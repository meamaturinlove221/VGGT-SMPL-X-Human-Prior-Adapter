"""CPU按四个真实TRAIN人体轮廓切开原GW连片，不重建人体。"""
from pathlib import Path
import json
import cv2,numpy as np,trimesh
from PIL import Image,ImageDraw

from research.dna_gw40_dngw0019_replay_20261125.paths import OFFICIAL as BASE, RESULTS
OUT=BASE/'human_body_only'
SOURCE=BASE/'GW_OFFICIAL_DIRECT_DEFAULT/mesh_ours_2pivots_post_texture_refined_999.ply'
INPUT = RESULTS/'DNA_GAUSSIAN_TEACHER_EXEC_20261102/inputs/DNA0019_INPUT.json'
from extract_gw_human_display import save_ply,project,points_image

def view_panel(v,f,rgb,row,size):
    """只用CPU画原彩色三角面，不导入训练环境。"""
    uv,z=project(v,row);im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im)
    ids=np.flatnonzero((z[f]>0).all(1));colors=np.rint(rgb[f].mean(1)).astype(np.uint8)
    for i in ids[np.argsort(z[f[ids]].mean(1))[::-1]]:
        d.polygon([tuple(p) for p in uv[f[i]]],fill=tuple(colors[i]))
    return im

def colored_mesh_panel(v,f,rgb,center,scale,angle,label):
    """四角同尺度完整范围，深度排序只决定展示顺序。"""
    p=(v-center)[:,[0,2,1]]*np.array([1,1,-1]);r=np.deg2rad(angle);c=np.cos(r);s=np.sin(r)
    horizontal=p[:,0]*c+p[:,1]*s;depth=-p[:,0]*s+p[:,1]*c;half=float(scale)*.53
    uv=np.stack(((horizontal/(2*half)+.5)*499,30+(.5-p[:,2]/(2*half))*499),axis=-1)
    colors=np.rint(rgb[f].mean(1)).astype(np.uint8);im=Image.new('RGB',(500,530),'white');d=ImageDraw.Draw(im)
    for i in np.argsort(depth[f].mean(1)):d.polygon([tuple(x) for x in uv[f[i]]],fill=tuple(colors[i]))
    d.rectangle((0,0,500,28),fill='white');d.text((6,5),label,fill='black');return im

def supported(points,rows):
    """四个明确人体mask中至少三个支持，容许单视边缘不一致。"""
    count=np.zeros(len(points),np.uint8)
    for row in rows:
        uv,z=project(points,row);xy=np.rint(uv).astype(int);h,w=row['_mask'].shape
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ii=np.flatnonzero(valid);ok=np.zeros(len(points),bool)
        ok[ii]=row['_mask'][xy[ii,1],xy[ii,0]];count+=ok
    return count>=3

def main():
    """一次固定四角轮廓规则，原面内7个位置都需支持。"""
    OUT.mkdir(exist_ok=True)
    info=json.loads(INPUT.read_text(encoding='utf8'))
    rows=[next(r for r in info['rows'] if r['camera_id']==i) for i in (2,26,11,35)]
    for row in rows:
        assert row['camera_id'] in info['train_ids']
        mask=np.asarray(Image.open(row['mask']).convert('L'))>=128
        row['_mask']=cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0
    m=trimesh.load(SOURCE,process=False,maintain_order=True)
    v=np.asarray(m.vertices);f=np.asarray(m.faces);rgb=np.asarray(m.visual.vertex_colors)[:,:3]
    vertex_ok=supported(v,rows);keep=vertex_ok[f].all(1)
    # 巨三角即使端点落在人身上，跨过轮廓空隙也不能保留。
    tri=v[f];samples=[tri.mean(1),(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2]
    for p in samples:keep &= supported(p,rows)
    fi=np.flatnonzero(keep);vi=np.unique(f[keep]);mapping=np.full(len(v),-1,np.int32);mapping[vi]=np.arange(len(vi))
    vv=v[vi];ff=mapping[f[keep]];cc=rgb[vi]
    mesh=OUT/'GW_HUMAN_BODY_ONLY_DISPLAY.ply';point=OUT/'GW_HUMAN_BODY_ONLY_POINTS.ply'
    save_ply(mesh,vv,cc,ff);save_ply(point,vv,cc)
    np.save(OUT/'ORIGINAL_VERTEX_INDICES.npy',vi);np.save(OUT/'ORIGINAL_FACE_INDICES.npy',fi);np.save(OUT/'REMOVED_FACE_INDICES.npy',np.flatnonzero(~keep))
    center=(vv.min(0)+vv.max(0))/2;scale=np.ptp(vv,axis=0).max();overview=Image.new('RGB',(2000,530),'white')
    for j,angle in enumerate((0,90,180,270)):
        overview.paste(colored_mesh_panel(vv,ff,cc,center,scale,angle,f'BODY ONLY ALL FACES {angle}'),(j*500,0))
    overview.save(OUT/'BODY_ONLY_FOUR_ANGLES.png')
    sheet=Image.new('RGB',(1800,2400),'white');d=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        raw=Image.open(row['rgb']).convert('RGB');ma=np.asarray(Image.open(row['mask']).convert('L'))>=128;y,x=np.where(ma)
        box=(max(0,x.min()-30),max(0,y.min()-30),min(raw.width,x.max()+31),min(raw.height,y.max()+31))
        arr=np.asarray(raw).copy();contours,_=cv2.findContours(ma.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,contours,-1,(0,255,0),1)
        panels=[Image.fromarray(arr),view_panel(vv,ff,cc,row,raw.size),points_image(vv,cc,row,raw.size)]
        for j,im in enumerate(panels):
            im=im.crop(box);im.thumbnail((590,560));sheet.paste(im,(j*600+(600-im.width)//2,i*600+30));d.text((j*600+5,i*600+5),f'cam{row["camera_id"]} '+['TRAIN RGB original mask','BODY ONLY ORIGINAL FACES','SAME MESH VERTICES'][j],fill='black')
    sheet.save(OUT/'TRAIN_2_26_11_35_BODY_COMPARE.png')
    # 回读只检查交付文件是否丢失颜色、原坐标或三角面索引。
    reread=trimesh.load(mesh,process=False,maintain_order=True)
    readback=dict(XYZ=np.array_equal(reread.vertices,vv.astype(np.float32)),RGB=np.array_equal(np.asarray(reread.visual.vertex_colors)[:,:3],cc),faces=np.array_equal(reread.faces,ff))
    assert all(readback.values())
    rec=dict(status='DISPLAY_BODY_ONLY_PENDING_VISUAL_REVIEW',source=str(SOURCE),source_vertices=len(v),source_faces=len(f),mesh=str(mesh),mesh_bytes=mesh.stat().st_size,points=str(point),points_bytes=point.stat().st_size,vertices=len(vv),faces=len(ff),removed_faces=int((~keep).sum()),rule='Original face subset only. TRAIN cams2/26/11/35 original 1224x1024 RGB+K/E+mask. Fixed mask>=128, fixed8px square rim. Require >=3of4 positive-Z/in-frame/foreground support at all three vertices, all three edge midpoints and face centroid. Single-view disagreement allowed to preserve shoes/hair. No 39mask intersection; no largest-component deletion.',selection=[{k:r[k] for k in ('camera_id','rgb','mask','K','E')} for r in rows],vertex_index='ORIGINAL_VERTEX_INDICES.npy',face_index='ORIGINAL_FACE_INDICES.npy',removed_face_index='REMOVED_FACE_INDICES.npy',source_only_cube=[info['bbox_min'],info['bbox_max']],cube_used_to_delete=False,camera17='MASK_SEMANTICS_UNRESOLVED,not used',held_Kinect_scan_used=False,training=0,GPU=0,reconstruction=0,smoothing_filling_ICP=False,threshold_sweep=False,teacher_permission='NOT_GRANTED; display-only cleanup',readback=readback,visual_review='PENDING_ACTUAL_VIEW_IMAGE')
    (OUT/'BODY_ONLY_RECEIPT.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:rec[k] for k in ('mesh','mesh_bytes','vertices','faces','removed_faces','readback')}))

if __name__=='__main__':main()
