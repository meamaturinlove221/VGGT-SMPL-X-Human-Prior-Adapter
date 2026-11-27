"""仅CPU按既有TRAIN人体mask选择GW原网格顶点；不重建、不授予教师许可。"""
from pathlib import Path
import json
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageDraw

from research.dna_gw40_dngw0019_replay_20261125.paths import RESULTS

BASE=RESULTS/'DNA_GAUSSIAN_MESH_MENTOR_20261102'
SOURCE=BASE/'mtet_interface_repair_20261102/model/mesh_ours_2pivots.ply'
OUT=BASE/'human_display'
OLD=RESULTS/'DNA_GAUSSIAN_TEACHER_EXEC_20261102/inputs/DNA0019_INPUT.json'

def project(v,row):
    """原OpenCV世界到相机投影，保留正深度判定。"""
    E=np.array(row['E']);K=np.array(row['K']);c=v@E[:3,:3].T+E[:3,3]
    q=c@K.T
    return q[:,:2]/q[:,2,None],c[:,2]

def points_image(v,rgb,row,wh):
    """CPU最近点显示；不以显示首面反过来选择几何。"""
    uv,z=project(v,row);xy=np.rint(uv).astype(int);w,h=wh
    good=(z>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    ids=np.flatnonzero(good);ids=ids[np.argsort(z[ids])[::-1]]
    im=np.full((h,w,3),245,np.uint8)
    im[xy[ids,1],xy[ids,0]]=rgb[ids]
    return Image.fromarray(im)

def save_ply(path,v,rgb,faces=None):
    """输出原float32坐标及原RGB，点文件显式零面。"""
    f=np.empty((0,3),np.int32) if faces is None else np.asarray(faces,np.int32)
    records=np.empty(len(v),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):records[k]=v[:,i]
    for i,k in enumerate(('red','green','blue')):records[k]=rgb[:,i]
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n'
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'));records.tofile(stream)
        if len(f):
            a=np.empty(len(f),dtype=[('n','u1'),('indices','<i4',(3,))]);a['n']=3;a['indices']=f;a.tofile(stream)

def main():
    """一次固定规则筛选并保存可审计索引和实际多角预览。"""
    OUT.mkdir(exist_ok=True)
    info=json.loads(OLD.read_text(encoding='utf8'))
    m=trimesh.load(SOURCE,process=False,maintain_order=True)
    v=np.asarray(m.vertices);rgb=np.asarray(m.visual.vertex_colors)[:,:3];faces=np.asarray(m.faces)
    cube=((v>=np.array(info['bbox_min']))&(v<=np.array(info['bbox_max']))).all(1)
    keep=cube.copy();rows=[];perview=[];support=np.zeros(len(v),np.int16)
    # camera17语义未决，保留冻结输入但不让它决定删点。
    for row in info['rows']:
        if row['camera_id'] not in info['train_ids']:continue
        if row['camera_id']==17:continue
        mask=np.asarray(Image.open(row['mask']).convert('L'))>=128
        expanded=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
        uv,z=project(v,row);xy=np.rint(uv).astype(int);h,w=mask.shape
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        fg=np.zeros(len(v),bool);ids=np.flatnonzero(valid);fg[ids]=expanded[xy[ids,1],xy[ids,0]]
        keep &= fg;support+=fg;rows.append(row)
        perview.append(dict(camera_id=row['camera_id'],positive_in_frame=int(valid.sum()),foreground_supported=int(fg.sum())))
    ids=np.flatnonzero(keep);vv=v[ids];cc=rgb[ids]
    point=OUT/'GW_HUMAN_BODY_DISPLAY_POINTS.ply';save_ply(point,vv,cc)
    np.save(OUT/'ORIGINAL_VERTEX_INDICES.npy',ids)
    fmap=np.full(len(v),-1,np.int32);fmap[ids]=np.arange(len(ids));ff=faces[keep[faces].all(1)]
    mesh=OUT/'GW_HUMAN_BODY_DISPLAY_ORIGINAL_FACES.ply';save_ply(mesh,vv,cc,fmap[ff])
    # 对照原照片和原mask，不使用held或Kinect/scan选点。
    cams=(2,26,11,35);sheet=Image.new('RGB',(1500,720*len(cams)),(255,255,255));draw=ImageDraw.Draw(sheet)
    for i,cam in enumerate(cams):
        row=next(x for x in rows if x['camera_id']==cam);raw=Image.open(row['rgb']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        ma=np.asarray(mask)>0;y,x=np.where(ma);bounds=(max(0,x.min()-35),max(0,y.min()-35),min(raw.width,x.max()+36),min(raw.height,y.max()+36))
        arr=np.asarray(raw).copy();contours,_=cv2.findContours(ma.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,contours,-1,(0,255,0),1)
        panels=[('TRAIN RGB + fixed mask',Image.fromarray(arr)),('Original GW vertices',points_image(v,rgb,row,raw.size)),('Selected original GW points',points_image(vv,cc,row,raw.size))]
        for j,(label,im) in enumerate(panels):
            im=im.crop(bounds);im.thumbnail((490,680));sheet.paste(im,(j*500+(500-im.width)//2,i*720+35));draw.text((j*500+8,i*720+8),f'cam{cam} {label}',fill=(0,0,0))
    sheet.save(OUT/'GW_HUMAN_BODY_PREVIEW_TRAIN_4ANGLES.png')
    # 回读检测写出损失；若失败则不能交付保真声明。
    reread=trimesh.load(point,process=False,maintain_order=True)
    exact_xyz=np.array_equal(np.asarray(reread.vertices),vv.astype(np.float32).astype(float));exact_rgb=np.array_equal(np.asarray(reread.colors)[:,:3],cc)
    assert exact_xyz and exact_rgb and len(reread.vertices)==len(ids)
    rec=dict(status='DISPLAY_ONLY_TRAIN_SILHOUETTE_FILTERED_GW_VERTICES',source=str(SOURCE),source_vertices=len(v),source_faces=len(faces),source_components=12,independent_human_component=False,component_reason='Main component contains both human and external thin sheets; selecting largest cannot isolate the body.',output_points=str(point),retained_vertices=len(ids),removed_vertices=len(v)-len(ids),output_faces=0,optional_original_faces=len(ff),output_mesh=str(mesh),source_cube_retained=int(cube.sum()),source_cube_min=info['bbox_min'],source_cube_max=info['bbox_max'],rule='Original vertices inside existing source-only cube AND positive cameraZ/in frame/inside original TRAIN foreground in every one of 39 semantically usable TRAIN masks. Fixed mask>=128, 2px square dilation at 1224x1024 for raster quantization and minor boundary tolerance; no parameter sweep.',frozen_train_ids=info['train_ids'],selection_train_ids=[r['camera_id'] for r in rows],excluded_camera17='MASK_SEMANTICS_UNRESOLVED; unchanged and not used to delete vertices',held_Kinect_scan_used=False,per_train_view=perview,bounds_m=[vv.min(0).tolist(),vv.max(0).tolist()],coordinates='identity calibrated world metres, original float32 XYZ and uint8 RGB copied',record_semantics='Vertices of GW MTet mesh; NOT Gaussian centres and NOT VGGT RAW',readback=dict(vertex_count=len(reread.vertices),face_count=0,exact_original_xyz=exact_xyz,exact_original_RGB=exact_rgb),smoothing_filling_reconstruction=False,training_updates=0,GPU_used=False,threshold_sweep=False,limitations='Silhouette visual hull restriction cannot remove internal or silhouette-consistent sheets, separate coincident layers, or restore missing hair/hands/shoes. Intersection can discard original boundary vertices where masks/calibration disagree. Points display must not be called complete or validated human reconstruction, nor teacher-approved.',preview=str(OUT/'GW_HUMAN_BODY_PREVIEW_TRAIN_4ANGLES.png'),preview_visual_review='PENDING_ACTUAL_VIEW_IMAGE',files=[dict(path=str(p),bytes=p.stat().st_size) for p in (point,mesh,OUT/'ORIGINAL_VERTEX_INDICES.npy',OUT/'GW_HUMAN_BODY_PREVIEW_TRAIN_4ANGLES.png')])
    (OUT/'GW_HUMAN_BODY_DISPLAY_RECEIPT.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'README.md').write_text('# GW人体范围展示点云\n\nGW_HUMAN_BODY_DISPLAY_POINTS.ply是GW原MTet网格顶点筛选，零面彩色PLY；不是Gaussian中心、VGGT RAW或新重建。原坐标/RGB逐记录保留，完整原GW未改。\n\n一次固定筛选：既有source-only cube，39路语义明确TRAIN正深度/视野/前景交集，原mask>=128并固定膨胀2像素。camera17保持MASK_SEMANTICS_UNRESOLVED，未作为删点依据。未使用held/Kinect/scan误差、未阈值扫描、未训练/GPU/平滑/补洞。ORIGINAL_VERTEX_INDICES.npy对应原顶点索引。可选mesh只保留三顶点都通过的原面。\n\n主组件身体与片层连通，不能通过选最大组件可靠提纯。此展示可去掉人体轮廓外污染，但人体轮廓内重叠片层仍可能存在；mask交集也可能损失原边缘。不得称完整纯人体重建，不升级教师许可。照片/原点/筛选点四角预览及详细计数见receipt。\n',encoding='utf8')
    print(json.dumps({k:rec[k] for k in ('status','retained_vertices','removed_vertices','optional_original_faces','readback','files')}),flush=True)

if __name__=='__main__':main()
