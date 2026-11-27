"""0102/0000 的 BUILD34 原相机、固定标定 SIFT 和居中 GW 输入。"""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pycolmap
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, COLMAP, RESULTS
OUT=RESULTS/"DNA_GW_SECOND_IDENTITY_0102_20261123"
RUN=OUT/"GW_0102_34BUILD_6AUX_8EVAL"
CASE=ARTIFACTS/"DNA_MULTISUBJECT_REPAIR_20261024/inputs/DNA_RENDERING__0102__0102_02__0000"
SIFT=RUN/"known_pose_build34"
INPUT=RUN/"official_input_build34"
FIX=RESULTS/"DNA_GW_GEOMETRY_STUDENT_20261103/teacher_neighbor_fix"
VENDOR=FIX/"vendor/GaussianWrapping/gaussian_wrapping"
AUX={3,5,18,27,32,36}
EVAL={1,8,12,23,28,33,38,44}
STUDENT=[2,11,17,26,35,41]


def read(p):
    """读取冻结相机与原始输入回执。"""
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save(p,x):
    """保存已执行的调用和当前资产。"""
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")


def csvfile(p,rows):
    """保存逐相机、逐配对或逐视角测量。"""
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def get_rows():
    """以原 TRAIN40/HELD8 为准，取出指定 AUX6。"""
    src=read(CASE/"RECOVERED_READ48_RECEIPT.json")
    split=read(CASE/"SPLIT40_8.json")
    build=set(split["train"])-AUX
    assert len(build)==34 and set(range(48))-set(split["train"])==EVAL
    return src,[r for r in src["records"] if r["camera_id"] in build]


def camera_manifest():
    """核实保存 48 张 RGB 与 Mask，写出角色及修正邻居预检。"""
    src,rows=get_rows()
    cams=[]
    for r in src["records"]:
        i=r["camera_id"]; rgb=np.asarray(Image.open(r["rgb"]).convert("RGB"));m=np.asarray(Image.open(r["mask"]).convert("L"))
        assert rgb.shape[:2]==m.shape==tuple(r["output_hw"]) and np.isfinite(r["K"]).all()
        cams.append(dict(camera=i,role="EVAL" if i in EVAL else "AUX" if i in AUX else "BUILD",student_slot=STUDENT.index(i) if i in STUDENT else "",
                         original_frame=0,source_rgb=r["rgb"],source_mask=r["mask"],native_rgb_smc=src["main_archive"],rgb_key=r["rgb_key"],annotation_smc=src["annotation_archive"],
                         mask_key=r["mask_key"],rgb_width=rgb.shape[1],rgb_height=rgb.shape[0],nonzero_mask=int(np.count_nonzero(m)),K=json.dumps(r["K"]),
                         K_coordinate="prepared 1224x1024 undistorted OpenCV",K_native=json.dumps(r["native_K"]),D_native=json.dumps(r["native_D"]),
                         native_width=r["native_hw"][1],native_height=r["native_hw"][0],E=json.dumps(r["E"]),RT_c2w=json.dumps(r["official_RT_c2w"]),pixel_bridge=json.dumps(r["native_to_input"])))
    csvfile(OUT/"0102_CAMERA_ROLE_MANIFEST.csv",cams)
    rt=np.linalg.inv(np.array([r["E"] for r in rows]));center=rt[:,:3,3];ray=rt[:,:3,2];ray/=np.linalg.norm(ray,axis=1,keepdims=True)
    radius=float(1.1*np.linalg.norm(center-center.mean(0),axis=1).max())
    dist=np.linalg.norm(center[:,None]-center[None],axis=2);ang=np.rad2deg(np.arccos(np.clip(ray@ray.T,-1,1)));pairs=[]
    for i,r in enumerate(rows):
        order=np.lexsort((ang[i],dist[i]));keep=(ang[i,order]<30)&(dist[i,order]>.002*radius)&(dist[i,order]<.3*radius)
        chosen=order[keep][:8]
        for j in chosen:
            pairs.append(dict(camera=r["camera_id"],neighbor=rows[j]["camera_id"],distance_m=dist[i,j],angle_deg=ang[i,j],radius_m=radius))
        if not len(chosen):
            pairs.append(dict(camera=r["camera_id"],neighbor="",distance_m="",angle_deg="",radius_m=radius))
    csvfile(OUT/"GW_0102_NEIGHBOR_GRAPH.csv",pairs)
    empty=[p["camera"] for p in pairs if p["neighbor"]==""]
    (OUT/"GW_0102_BUILD34_PREFLIGHT.md").write_text(f"# BUILD34 原始相机预检\n\n实读 48 张同帧 RGB 和 48 张 Mask。34 BUILD/6 AUX/8 EVAL；固定校正每相机邻居上限 8、角度小于 30°、距离为半径的 (0.002,0.3)。BUILD 半径 {radius:.6f} m，{34-len(empty)}/34 路有候选，{len(pairs)-len(empty)} 个有向配对，无邻居相机 {empty}。运行时接受配对由训练日志另记。\n",encoding="utf-8")
    save(RUN/"RUN_CONTRACT.json",dict(sample=src["sample_id"],BUILD=[r["camera_id"] for r in rows],AUX=sorted(AUX),EVAL=sorted(EVAL),student_slots=STUDENT,source=str(CASE),
                                     sift="real BUILD RGB, exhaustive matching, fixed calibrated PINHOLE and fixed frames",training="fresh independent BUILD34 SIFT init, official ours, per-camera corrected neighbor cap",
                                     vendor=str(VENDOR),input=str(INPUT),output=str(RUN/"model"),nominal_iterations=30000,texture_steps=1000,body_only_cleanup=False))


def invoke(command,env,ledger):
    """执行一次指定的 COLMAP 阶段，原命令、返回值和 stdout 留证。"""
    name=command[1]
    entry=dict(stage=name,cmd=command,start=datetime.now().astimezone().isoformat())
    ledger.append(entry);save(RUN/"SIFT_CALLS.json",ledger)
    with (SIFT/f"{name}.log").open("wb") as f:
        res=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    entry.update(returncode=res.returncode,end=datetime.now().astimezone().isoformat());save(RUN/"SIFT_CALLS.json",ledger)
    if res.returncode:raise RuntimeError(f"{name}: {res.returncode}")


def sift(rows):
    """使用 3.13 真实数据库 rig/frame ID 写出固定相机模型，保留固定相机的点内优化。"""
    paths=[SIFT/s for s in ("images","known_model","sparse/0")];images,known,sparse=paths
    for p in paths:p.mkdir(parents=True,exist_ok=True)
    dbpath=SIFT/"database.db";assert not dbpath.exists()
    db=pycolmap.Database.open(str(dbpath));cl=[];il=[];rigs=[];frames=[]
    for r in rows:
        cid=r["camera_id"];uid=cid+1;name=f"cam_{cid:02d}.png";shutil.copyfile(r["rgb"],images/name)
        h,w=r["output_hw"];k=np.array(r["K"]);params=[k[0,0],k[1,1],k[0,2]+.5,k[1,2]+.5]
        db.write_camera(pycolmap.Camera(camera_id=uid,model="PINHOLE",width=w,height=h,params=params),use_camera_id=True)
        db.write_image(pycolmap.Image(image_id=uid,camera_id=uid,name=name),use_image_id=True)
        e=np.array(r["E"]);q=Rotation.from_matrix(e[:3,:3]).as_quat()[[3,0,1,2]]
        pose=" ".join(format(x,".17g") for x in [*q,*e[:3,3]])
        cl.append(f"{uid} PINHOLE {w} {h} "+" ".join(format(x,".17g") for x in params))
        il.append(f"{uid} {pose} {uid} {name}\n")
    db.close()
    for file,lines in [("cameras",cl),("images",il),("points3D",[])]: (known/f"{file}.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    env=dict(os.environ);env["PATH"]=str(COLMAP.parent)+os.pathsep+env["PATH"];ledger=[]
    common=["--database_path",str(dbpath)]
    invoke([str(COLMAP),"feature_extractor",*common,"--image_path",str(images),"--ImageReader.camera_model","PINHOLE","--FeatureExtraction.use_gpu","1"],env,ledger)
    invoke([str(COLMAP),"exhaustive_matcher",*common,"--FeatureMatching.use_gpu","1"],env,ledger)
    db=pycolmap.Database.open(str(dbpath));mapping={rig.ref_sensor_id.id:rig.rig_id for rig in db.read_all_rigs()}
    for r,line in zip(rows,il):
        uid=r["camera_id"]+1;pose=" ".join(line.split()[1:8]);rig=mapping[uid]
        rigs.append(f"{rig} 1 CAMERA {uid}");frames.append(f"{uid} {rig} {pose} 1 CAMERA {uid} {uid}")
    for file,lines in [("rigs",rigs),("frames",frames)]: (known/f"{file}.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    m=pycolmap.Reconstruction(str(known))
    existing={f.frame_id for f in db.read_all_frames()}
    for f in m.frames.values():
        if f.frame_id not in existing:db.write_frame(f,use_frame_id=True)
    db.close()
    invoke([str(COLMAP),"point_triangulator",*common,"--image_path",str(images),"--input_path",str(known),"--output_path",str(sparse),"--refine_intrinsics","0","--Mapper.fix_existing_frames","1","--Mapper.ba_refine_sensor_from_rig","0"],env,ledger)


def center(rows):
    """同步变换原 RGB/alpha、K 与已有 tracks，保留全幅和全部 SIFT 点。"""
    m=pycolmap.Reconstruction(str(SIFT/"sparse/0"));assert m.num_points3D()>0 and m.num_reg_images()==34
    known=pycolmap.Reconstruction(str(SIFT/"known_model"))
    width,height=[4*math.ceil(max(2*max(r["K"][j][2],r["output_hw"][1-j]-1-r["K"][j][2])+3 for r in rows)/4) for j in (0,1)]
    sparse=INPUT/"sparse/0";images=INPUT/"images";sparse.mkdir(parents=True,exist_ok=True);images.mkdir(exist_ok=True)
    stats=[];dE=dK=0.0
    for r in rows:
        cid=r["camera_id"]+1;cam=m.camera(cid);im=m.image(cid)
        dK=max(dK,float(np.abs(cam.params-known.camera(cid).params).max()))
        dE=max(dE,float(np.abs(im.cam_from_world().matrix()-np.array(r["E"])[:3]).max()))
        k=np.array(r["K"]);delta=np.array([(width-1)/2,(height-1)/2])-k[:2,2]
        mat=np.array([[1,0,delta[0]],[0,1,delta[1]]]);rgba=np.asarray(Image.open(r["rgb"]).convert("RGBA")).copy()
        rgba[...,3]=np.asarray(Image.open(r["mask"]).convert("L"));centered=cv2.warpAffine(rgba,mat,(width,height),flags=cv2.INTER_LINEAR)
        Image.fromarray(centered).save(images/im.name)
        cam.width=width;cam.height=height;cam.params=np.array([k[0,0],k[1,1],width/2,height/2])
        for p in im.points2D:p.xy=p.xy+delta
        stats.append(dict(camera=r["camera_id"],image=im.name,original_K=r["K"],E=r["E"],centered_K=[[k[0,0],0,(width-1)/2],[0,k[1,1],(height-1)/2],[0,0,1]],
                          offset_xy=delta.tolist(),stored_tracks=len(im.points2D)))
    assert dK<1e-8 and dE<1e-6,(dK,dE)
    m.write(str(sparse));m.export_PLY(str(OUT/"GW_0102_SIFT_POINTS.ply"))
    save(RUN/"CENTERED_BUILD34.json",dict(width=width,height=height,rows=stats,max_K_delta=dK,max_E_delta=dE,triangulated=m.num_points3D(),
                                        initial_points_bin=str(sparse/"points3D.bin"),world_unit="original calibrated metres"))
    print("READY",m.num_points3D(),width,height,flush=True)


if __name__=="__main__":
    RUN.mkdir(parents=True,exist_ok=True);camera_manifest()
    _,rows=get_rows();sift(rows);center(rows)
