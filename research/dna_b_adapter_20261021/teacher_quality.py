"""教师全组件mesh的独立留出轮廓与原六源Kinect首面评价。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from matched_raw import write_json,write_csv,write_ply
from matched_raw_figures import make_figures

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/thuman20/DNA_TEACHER_COMPARE_20261023'


def vertex_normals(vertices,faces):
    """由全组件原面计算顶点法线，不翻面、不平滑几何或删组件。"""
    triangles=vertices[faces]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    summed=np.zeros_like(vertices)
    for corner in range(3):np.add.at(summed,faces[:,corner],normal)
    return summed/np.maximum(np.linalg.norm(summed,axis=1,keepdims=True),1e-12)


@torch.no_grad()
def render(vertices,faces,normals,K,E,height,width,context):
    """以真实相机和整数像素中心获取全部mesh的最近首面。"""
    intrinsic=torch.as_tensor(K,dtype=torch.float32,device='cuda')
    extrinsic=torch.as_tensor(E,dtype=torch.float32,device='cuda')
    camera=vertices@extrinsic[:3,:3].T+extrinsic[:3,3]
    near,far=.01,100.
    clip=torch.empty((1,len(vertices),4),device='cuda')
    clip[0,:,0]=2*intrinsic[0,0]/width*camera[:,0]+2*intrinsic[0,1]/width*camera[:,1]+(2*(intrinsic[0,2]+.5)/width-1)*camera[:,2]
    clip[0,:,1]=2*intrinsic[1,1]/height*camera[:,1]+(2*(intrinsic[1,2]+.5)/height-1)*camera[:,2]
    clip[0,:,2]=(far+near)/(far-near)*camera[:,2]-2*far*near/(far-near)
    clip[0,:,3]=camera[:,2]
    raster,_=dr.rasterize(context,clip,faces,resolution=(height,width))
    attributes=torch.cat([camera[:,2:3],normals@extrinsic[:3,:3].T],dim=1)[None]
    interpolated,_=dr.interpolate(attributes,raster,faces)
    valid=raster[0,:,:,3]>0
    depth=interpolated[0,:,:,0]
    normal=interpolated[0,:,:,1:4]
    normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return dict(depth=depth.cpu().numpy().astype(np.float32),valid=valid.cpu().numpy(),
                normal=normal.cpu().numpy().astype(np.float32))


def overlay(path,rgb,mask,predicted,label):
    """真实照片上叠加观测绿边与预测红边，不生成或修改源照片。"""
    canvas=rgb.copy()
    for binary,color in [(mask,(0,220,0)),(predicted,(0,0,255))]:
        contours,_=cv2.findContours(binary.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,color,1)
    cv2.putText(canvas,label,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.imwrite(str(path),canvas)


def quality(sequence,frame,mesh_path,phase,backend,folder=None):
    """分报轮廓错误、可测首面误差和缺失，未图审前不授予教师通过。"""
    folder=Path(folder) if folder is not None else OUT/'teacher'/f'{sequence}_{frame:04d}'
    receipt=json.loads((folder/'INPUT_RECEIPT.json').read_text(encoding='utf-8'))
    out=folder/'quality'/phase;out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(mesh_path,process=False,maintain_order=True)
    vertices=np.asarray(mesh.vertices,np.float32);faces=np.asarray(mesh.faces,np.int32)
    if not len(vertices) or not len(faces):raise ValueError('Empty teacher mesh is a failed teacher, not a target.')
    normals=vertex_normals(vertices,faces)
    np.savez_compressed(out/'ALL_COMPONENTS_MESH.npz',vertices=vertices,faces=faces,normals=normals)
    mesh.export(out/'ALL_COMPONENTS_MESH.ply')
    v=torch.as_tensor(vertices,device='cuda');f=torch.as_tensor(faces,device='cuda');n=torch.as_tensor(normals,device='cuda')
    context=dr.RasterizeCudaContext() if backend=='cuda' else dr.RasterizeGLContext()
    heldrows=[]
    render_calls_started=0
    render_calls_completed=0
    for record in receipt['heldout']:
        mask=cv2.imread(record['mask'],cv2.IMREAD_GRAYSCALE)>0
        rgb=cv2.imread(record['rgb'],cv2.IMREAD_COLOR)
        render_calls_started+=1
        print(json.dumps(dict(stage='QA_RENDER_STARTED',phase=phase,kind='HELD',camera_id=record['camera_id'],call=render_calls_started)),flush=True)
        rendered=render(v,f,n,record['K'],record['E'],*mask.shape,context)
        render_calls_completed+=1
        print(json.dumps(dict(stage='QA_RENDER_COMPLETED',phase=phase,call=render_calls_completed)),flush=True)
        pred=rendered['valid'];tp=int((pred&mask).sum());fn=int((~pred&mask).sum());fp=int((pred&~mask).sum())
        row=dict(camera_id=record['camera_id'],TP=tp,FN=fn,FP=fp,GT_foreground=int(mask.sum()),prediction_foreground=int(pred.sum()),
                 FN_rate=fn/int(mask.sum()),FP_rate=fp/max(int(pred.sum()),1),IoU=tp/max(tp+fn+fp,1),
                 scope='Heldout official same-frame masks; no neighbor or teacher optimization exposure')
        heldrows.append(row);np.savez_compressed(out/f'HELDOUT_{record["camera_id"]:02d}.npz',**rendered)
        overlay(out/f'HELDOUT_{record["camera_id"]:02d}_OVERLAY.png',rgb,mask,pred,f'{phase} held cam {record["camera_id"]}: observation green, prediction red')
    sample=f'DNA_RENDERING/{sequence[:4]}/{sequence}/{frame:04d}'
    input_dir=ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021/inputs'/sample.replace('/','__')/'GIVEN_PRIOR'
    with np.load(input_dir/'INPUTS.npz') as z:
        K,E=z['K'][0],z['E'][0];masks=z['masks'][0].astype(bool);images=z['images'][0];prior=z['smpl_vertices'][0]
    with np.load(input_dir/'TARGETS.npz') as z:
        target=z['depth'][0];target_valid=z['depth_valid'][0].astype(bool)
    source=[];rows=[];worlds=[]
    for index in range(6):
        render_calls_started+=1
        print(json.dumps(dict(stage='QA_RENDER_STARTED',phase=phase,kind='ORIGINAL_518_SOURCE',view_index=index,call=render_calls_started)),flush=True)
        rendered=render(v,f,n,K[index],E[index],*masks[index].shape,context);source.append(rendered)
        render_calls_completed+=1
        print(json.dumps(dict(stage='QA_RENDER_COMPLETED',phase=phase,call=render_calls_completed)),flush=True)
        # 输出全部教师首面投影，不以照片mask或Kinect有效性裁剪候选几何。
        yy,xx=np.indices(masks[index].shape,dtype=np.float32)
        rays=np.stack([(xx-K[index,0,2])/K[index,0,0],(yy-K[index,1,2])/K[index,1,1],np.ones_like(xx)],axis=-1)
        camera=rays*rendered['depth'][...,None]
        worlds.append(np.einsum('hwi,ij->hwj',camera-E[index,:3,3],E[index,:3,:3],optimize=False))
        measured=target_valid[index]&masks[index]&np.isfinite(target[index])&(target[index]>0)
        valid=measured&rendered['valid']&np.isfinite(rendered['depth'])&(rendered['depth']>0)
        errors=np.abs(rendered['depth'][valid]-target[index][valid])
        rows.append(dict(view_index=index,measured_support=int(measured.sum()),rendered_measured_support=int(valid.sum()),
                         missing_first_surface_on_measured=int((measured&~valid).sum()),
                         camera_Z_abs_mean=float(errors.mean()) if len(errors) else None,
                         camera_Z_abs_P95=float(np.quantile(errors,.95)) if len(errors) else None,units='metres',
                         scope='Original student pixel K/E and Kinect valid source support; missing teacher surface not scored as zero'))
        rgb=np.uint8(np.clip(np.rint(images[index].transpose(1,2,0)*255),0,255))[:,:,::-1].copy()
        overlay(out/f'SOURCE_{index:02d}_OVERLAY.png',rgb,masks[index],rendered['valid'],f'{phase} original source {index}')
    np.savez_compressed(out/'SIX_SOURCE_CANDIDATE_TARGETS.npz',
                        depth=np.stack([r['depth'] for r in source]),valid=np.stack([r['valid'] for r in source]),
                        camera_normal=np.stack([r['normal'] for r in source]),K=K,E=E)
    world=np.stack(worlds);visible=np.stack([r['valid'] for r in source])
    rgb=np.uint8(np.clip(np.rint(images.transpose(0,2,3,1)*255),0,255))
    viewids=np.broadcast_to(np.arange(6,dtype=np.uint8)[:,None,None],visible.shape)
    np.savez_compressed(out/'ALL_VISIBLE_SOURCE_PROJECTIONS.npz',world_points=world,images=rgb,mask=visible,K=K,E=E)
    write_ply(out/'ALL_VISIBLE_SOURCE_RGB_POINTS.ply',world[visible],rgb[visible],viewids[visible])
    figures=make_figures([dict(name=f'{phase} teacher mesh first surfaces',points=world[visible],
                              colors=rgb[visible],view_id=viewids[visible])],out,prior)
    write_json(out/'FIGURE_INDEX.json',figures)
    write_csv(out/'HELDOUT_MASK_METRICS.csv',heldrows);write_csv(out/'KINECT_FIRST_SURFACE_METRICS.csv',rows)
    result=dict(status='CANDIDATE_AWAITING_COMPLETE_VISUAL_AND_QA_DECISION',sample=sample,sequence=sequence,frame=frame,phase=phase,mesh=str(mesh_path),
                vertices=len(vertices),faces=len(faces),all_components_preserved=True,renderer=backend,renderer_calls_started=render_calls_started,renderer_calls_completed=render_calls_completed,
                heldout=heldrows,source=rows,optimizer_updates=0,
                target_consumption_allowed=False,units='original calibrated metres; no ICP, body rescale, or GT fit',
                colored_points='All first-surface pixels of six projections, including outside photo masks; not a network depth RAW or complete scene reconstruction. Whole mesh separately preserved.',
                teacher_target_policy='Only accepted teacher may enter original B cube/sampling/DPSR oracle; this file is not an acceptance decision')
    write_json(out/'QUALITY_RECEIPT.json',result)
    print(json.dumps(dict(output=str(out),status=result['status'],heldout=heldrows,source=rows),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',default='0097_04');parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--mesh',required=True);parser.add_argument('--phase',choices=['VH','NCC'],required=True)
    parser.add_argument('--backend',choices=['gl','cuda'],default='gl')
    parser.add_argument('--folder',help='本轮实际教师目录；省略时复用历史0097输出规则')
    args=parser.parse_args();quality(args.sequence,args.frame,args.mesh,args.phase,args.backend,args.folder)
