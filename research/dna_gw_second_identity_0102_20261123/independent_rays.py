"""把冻结的四份原网格投向完全相同的 RGB 与 Kinect 原生射线。"""
import argparse
from itertools import chain

import cv2
import h5py
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont

from research.dna_gw40_dngw0019_replay_20261125.paths import MONO_FONT

from qa_protocol import (AUX,EVAL,FIG,KINECT,MODELS,OUT,QA,camera_frame,cameras,
                         grid_band,original_mesh,project_box,read,save,table)


def initialize():
    """按 PLY 原面序建立四个同世界的双面首交索引。"""
    scenes,mesh_info={},{}
    for name,path in MODELS.items():
        vertices,faces=original_mesh(path)
        used=np.unique(faces)
        if not len(used) or not np.isfinite(vertices[used]).all():
            raise ValueError(f"{name} 无有效米制三角形")
        scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
        scenes[name]=scene
        mesh_info[name]=dict(path=str(path),vertex_slots=len(vertices),vertices_used=len(used),faces=len(faces),
                             triangle_index="unmodified original PLY face row; includes all components",
                             world_min=vertices[used].min(0).tolist(),world_max=vertices[used].max(0).tolist())
    save(QA/"MESH_WORLD_SOURCES.json",mesh_info)
    return scenes


def make_rays(xy,k,d,rt):
    """原像素中心逆畸变，射线参数直接等于光轴 Z 米。"""
    pt=cv2.undistortPointsIter(np.asarray(xy,np.float64).reshape(-1,1,2),k,d,None,None,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-12)).reshape(-1,2)
    dirs=pt@rt[:3,:2].T+rt[:3,2]
    rays=np.empty((len(dirs),6),np.float32)
    rays[:,:3],rays[:,3:]=rt[:3,3],dirs
    return o3d.core.Tensor(rays),dirs


def intersect(scene,rays):
    """返回沿给定射线的最近正交点，不补洞。"""
    hit=scene.cast_rays(rays,nthreads=4)
    z=hit["t_hit"].numpy().copy()
    z[~np.isfinite(z)|(z<=0)]=np.nan
    face=hit["primitive_ids"].numpy().astype(np.int32)
    face[~np.isfinite(z)]=-1
    return z,face


def pixel_metrics(ref,hit,crop):
    """统计完整像素分母、交并、精确率与召回率。"""
    x0,y0,x1,y1=crop
    r,h=ref[y0:y1,x0:x1],hit[y0:y1,x0:x1]
    tp,fn,fp=int((r&h).sum()),int((r&~h).sum()),int((~r&h).sum())
    return dict(pixels=r.size,TP=tp,FN=fn,FP=fp,TN=r.size-tp-fn-fp,
                IoU=tp/(tp+fn+fp) if tp+fn+fp else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def row_depth(dz,stable,mask,band):
    """在固定实测全集计算命中误差和稳定邻域的近远类别。"""
    use=mask&np.isfinite(dz)
    h=use&stable
    error=dz[use]
    return dict(original_positive=int(mask.sum()),stable_reference=int((mask&stable).sum()),
                first_hit=int(use.sum()),miss_all=int((mask&~use).sum()),
                MAE_mm=float(np.abs(error).mean()) if len(error) else None,
                median_abs_mm=float(np.median(np.abs(error))) if len(error) else None,
                P95_abs_mm=float(np.percentile(np.abs(error),95)) if len(error) else None,
                median_signed_mm=float(np.median(error)) if len(error) else None,
                ray_band_agreement=int((h&(np.abs(dz)<=band)).sum()),
                model_nearer=int((h&(dz < -band)).sum()),model_farther=int((h&(dz > band)).sum()),
                stable_miss=int((stable&mask&~use).sum()),unstable_unknown=int((mask&~stable).sum()))


def native_measurements(sensor):
    """读取 Kinect 毫米、前景与原 K/D/RT，在原像素构造 3×3 稳定门。"""
    with KINECT.open("rb") as handle,h5py.File(handle,"r") as f:
        stem=f"Kinect/{sensor}"
        depth=f[stem+"/depth/0"][...]
        mask=cv2.imdecode(f[stem+"/mask/0"][...],cv2.IMREAD_UNCHANGED)
        positive=(mask>0 if mask.ndim==2 else (mask>0).any(axis=2))&(depth>0)
        cal=f[f"Calibration/Kinect/{sensor:02}"]
        k,d,rt=[cal[x][...].astype(float) for x in ("K","D","RT")]
    kernel=np.ones((3,3),np.uint8)
    valid3=cv2.erode((depth>0).astype("u1"),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    span=cv2.dilate(depth,kernel).astype(int)-cv2.erode(depth,kernel).astype(int)
    yy,xx=np.nonzero(positive)
    xy=np.column_stack((xx,yy))
    rays,dirs=make_rays(xy,k,d,rt)
    zm=depth[yy,xx].astype(float)*.001
    return xy,rays,dirs,zm,(valid3 & (span<=grid_band()*1000))[yy,xx],k,d,rt,depth.shape


def kinect_eval(scenes,regions):
    """保留八个原生设备每条实测射线的模型首面编号与全部残差。"""
    rows,parts=[],[]
    for sensor in range(8):
        xy,rays,dirs,zm,stable,k,d,rt,shape=native_measurements(sensor)
        world=rt[:3,3]+dirs*zm[:,None]
        masks=dict(FULL=np.ones(len(xy),bool))
        masks.update({name:np.all((bounds[0]<=world)&(world<=bounds[1]),axis=1) for name,bounds in regions.items()})
        payload=dict(sensor=sensor,pixel_xy=xy.astype("u2"),raw_depth_mm=np.rint(zm*1000).astype("u2"),
                     valid_3x3=stable,world_measured=world.astype("f4"),shape_hw=shape,K=k,D=d,RT=rt)
        values={}
        for model,scene in scenes.items():
            z,face=intersect(scene,rays)
            dz=(z-zm)*1000
            values[model]=dz
            payload[model+"_z_m"],payload[model+"_face"]=z,face
            for name,mask in masks.items():
                rows.append(dict(model=model,sensor=sensor,region=name,band_mm=grid_band()*1000,
                                 **row_depth(dz,stable,mask,grid_band()*1000)))
        np.savez_compressed(QA/f"KINECT{sensor}_FIRST_HITS.npz",**payload)
        parts.append((values,stable,masks))
    all_stable=np.concatenate([v[1] for v in parts])
    all_values={m:np.concatenate([v[0][m] for v in parts]) for m in scenes}
    common=np.logical_and.reduce([np.isfinite(a) for a in all_values.values()])
    for model,dz in all_values.items():
        for name in parts[0][2]:
            mask=np.concatenate([v[2][name] for v in parts])
            for tag,selection in [(name,mask),(name+"_COMMON4",mask&common)]:
                rows.append(dict(model=model,sensor="ALL",region=tag,band_mm=grid_band()*1000,
                                 **row_depth(dz,all_stable,selection,grid_band()*1000)))
    overlap=[]
    current=all_values["GW_OFFICIAL_FULL"]
    for other,other_dz in all_values.items():
        if other=="GW_OFFICIAL_FULL":
            continue
        for name in parts[0][2]:
            mask=np.concatenate([v[2][name] for v in parts])
            g=np.isfinite(current);ref=np.isfinite(other_dz)
            both=mask&g&ref
            for model,dz in (("GW_OFFICIAL_FULL",current),(other,other_dz)):
                error=np.abs(dz[both])
                overlap.append(dict(reference=other,model=model,region=name,fixed_denominator=int(mask.sum()),
                            both_hit=int(both.sum()),GW_only_hit=int((mask&g&~ref).sum()),
                            comparison_only_hit=int((mask&~g&ref).sum()),both_miss=int((mask&~g&~ref).sum()),
                            common_MAE_mm=float(np.mean(error)) if len(error) else None,
                            common_median_abs_mm=float(np.median(error)) if len(error) else None,
                            common_P95_abs_mm=float(np.percentile(error,95)) if len(error) else None))
    table(QA/"KINECT_PAIRWISE_COMMON_SUPPORT.csv",overlap)
    table(OUT/"GW_0102_KINECT_QA.csv",rows)
    return rows


def render(scenes,k,d,rt,h,w):
    """共用原像素方向分条投射四份网格。"""
    buffers={n:(np.empty((h,w),"f4"),np.empty((h,w),"i4")) for n in scenes}
    for start in range(0,h,96):
        end=min(h,start+96)
        xy=np.stack(np.meshgrid(np.arange(w),np.arange(start,end)),axis=-1)
        rays,_=make_rays(xy,k,d,rt)
        for n,scene in scenes.items():
            z,face=intersect(scene,rays)
            buffers[n][0][start:end]=z.reshape(end-start,w)
            buffers[n][1][start:end]=face.reshape(end-start,w)
    return buffers


def visual_panel(camera,role,bgr,ref,buffers,boxes):
    """用统一窗口显示原照片与每份模型的轮廓 TP、FN、FP。"""
    names=["ORIGINAL",*buffers]
    xy=np.argwhere(ref)
    x0,y0=xy.min(0)[::-1]
    x1,y1=xy.max(0)[::-1]+1
    windows={"FULL_FRAME":[0,0,ref.shape[1],ref.shape[0]],
             "SUBJECT":[x0,y0,x1,y1],"RIGHT_SHOE_ROBE":boxes["RIGHT_SHOE_ROBE"],
             "LOWER_HEM_HEIGHT":boxes["LOWER_HEM_HEIGHT"]}
    canvas=Image.new("RGB",(len(names)*246,1272),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(MONO_FONT),14)
    draw.text((8,4),f"0102_02 / 0000    {role} camera {camera:02}    native pixels    Green TP  Red FP  Blue FN",fill="black",font=font)
    for col,name in enumerate(names):
        img=bgr[:,:,::-1].copy()
        if name in buffers:
            hit=buffers[name][1]>=0
            for selection,color in [(hit&ref,[45,235,118]),(hit&~ref,[255,34,54]),(~hit&ref,[45,108,255])]:
                img[selection]=(.51*img[selection]+.49*np.array(color)).astype("u1")
        draw.text((col*246+3,27),name,fill="black",font=font)
        for i,(region,box) in enumerate(windows.items()):
            top=53+i*304
            patch=Image.fromarray(img).crop(tuple(map(int,box)))
            patch.thumbnail((238,274))
            canvas.paste(patch,(col*246+(246-patch.width)//2,top+24))
            draw.text((col*246+3,top),region,fill="black",font=font)
    canvas.save(FIG/f"{role}{camera:02}_FULL_AND_SHOE_COMPARISON.jpg",quality=93)


def camera_eval(scenes,regions):
    """候选冻结后才读取 EVAL 与 AUX 的原照和原 Mask。"""
    camera_book=cameras()
    evals,aux,raw_masks=[],[],[]
    for role,cam in chain((("EVAL",i) for i in EVAL),(("AUX",i) for i in AUX)):
        bgr,ref,k,d,rt,count=camera_frame(camera_book[cam])
        raw_masks.append(dict(camera=cam,role=role,**count))
        h,w=ref.shape
        boxes={"FULL":[0,0,w,h],**{n:project_box(bounds,k,d,rt,h,w) for n,bounds in regions.items()}}
        buffers=render(scenes,k,d,rt,h,w)
        for model,(z,face) in buffers.items():
            yy,xx=np.nonzero(face>=0)
            np.savez_compressed(QA/f"{role}{cam:02}_{model}.npz",xy=np.column_stack((xx,yy)).astype("u2"),
                     face_id=face[yy,xx],z_m=z[yy,xx],K=k,D=d,RT=rt,shape_hw=[h,w])
            for name,box in boxes.items():
                row=dict(model=model,camera=cam,role=role,region=name,xyxy=str(box),
                         **pixel_metrics(ref,face>=0,box))
                (evals if role=="EVAL" else aux).append(row)
        visual_panel(cam,role,bgr,ref,buffers,boxes)
        print(role,cam,"COMPLETE",flush=True)
    table(QA/"NATIVE_MASK_SEMANTICS.csv",raw_masks)
    for rows,filename in [(evals,"GW_0102_EVAL8.csv"),(aux,"AUX6_MASK_DIAGNOSTIC.csv")]:
        for model in scenes:
            for region in ["FULL",*regions]:
                source=[r for r in rows if r["model"]==model and r["region"]==region and r["camera"]!="ALL"]
                fields={key:sum(r[key] for r in source) for key in ["pixels","TP","FN","FP","TN"]}
                tp,fn,fp=(fields[x] for x in ["TP","FN","FP"])
                rows.append(dict(model=model,camera="ALL",role=source[0]["role"],region=region,xyxy="per-camera",
                            **fields,IoU=tp/(tp+fn+fp) if tp+fn+fp else None,
                            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
        table(OUT/filename,rows)


def main():
    """在收到四个冻结原文件后运行同协议 CPU 比较。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("part",choices=["all","kinect","cameras"])
    args=parser.parse_args()
    protocol=read(QA/"FROZEN_QA_PROTOCOL.json")
    scenes=initialize()
    if args.part in ["all","kinect"]:
        kinect_eval(scenes,protocol["region_boxes"])
    if args.part=="all":
        from exterior_support import main as exterior
        exterior(scenes)
        print("SIX_SOURCE_PHYSICAL_WITNESS_REVIEW_READY",flush=True)
    if args.part in ["all","cameras"]:
        camera_eval(scenes,protocol["region_boxes"])
    if args.part=="all":
        from mesh_views import mesh_panels,sensor_panel,source_panel
        mesh_panels(scenes)
        sensor_panel()
        source_panel()


if __name__=="__main__":
    main()
