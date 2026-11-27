"""官方同帧SMC到FastHuman NHR输入；训练与留出图像物理分离。"""
import argparse
import itertools
import json
from pathlib import Path
import cv2
import h5py
import numpy as np

from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, DATASETS

OUT = ARTIFACTS/'DNA_TEACHER_COMPARE_20261023'
DATA = DATASETS/'dna_rendering'
HELD=[5,12,19,27,34,42,49,56]


def dump(path,value):
    """只写本轮新输出，保存实际传感器和处理链。"""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def inverse_rt(c2w):
    """官方RT为c2w，显式刚体逆保持标定米单位。"""
    out=np.eye(4);out[:3,:3]=c2w[:3,:3].T
    out[:3,3]=-np.einsum('ij,j->i',out[:3,:3],c2w[:3,3],optimize=False)
    return out


def preprocess_official_view(bgr,mask,K,D):
    """同原像素链处理内存原图，供定向读取保留native后复用。"""
    height,width=bgr.shape[:2];w,h=1224,1024
    mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(width,height),cv2.CV_32FC1)
    rgb=cv2.resize(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR),(w,h),interpolation=cv2.INTER_AREA)
    foreground=cv2.resize(cv2.remap(mask,mx,my,cv2.INTER_NEAREST),(w,h),interpolation=cv2.INTER_NEAREST)>0
    sx,sy=w/width,h/height
    A=np.array([[sx,0,.5*(sx-1)],[0,sy,.5*(sy-1)],[0,0,1]])
    processed_K=np.einsum('ij,jk->ik',A,K,optimize=False)
    return rgb,foreground,processed_K,A


def export_view(main,annots,cid,frame,folder,index):
    """同一次undistort和resize成套处理真实RGB、mask与像素中心K。"""
    group='Camera_5mp' if cid<48 else 'Camera_12mp'
    rgbkey=f'{group}/{cid}/color/{frame}';maskkey=f'Mask/{cid}/mask/{frame}'
    bgr=cv2.imdecode(np.asarray(main[rgbkey]),cv2.IMREAD_COLOR)
    mask=cv2.imdecode(np.asarray(annots[maskkey]),cv2.IMREAD_UNCHANGED)
    if bgr is None or mask is None:raise ValueError(f'Actual image decode failed: camera {cid}, frame {frame}')
    if mask.ndim==3:mask=mask.max(2)
    if bgr.shape[:2]!=mask.shape:raise ValueError(f'RGB/mask native size mismatch: {cid}')
    cal=annots[f'Camera_Parameter/{cid:02d}'];K=np.asarray(cal['K'],float);D=np.asarray(cal['D'],float)
    c2w=np.asarray(cal['RT'],float);E=inverse_rt(c2w)
    height,width=bgr.shape[:2];w,h=1224,1024
    rgb,foreground,processed_K,A=preprocess_official_view(bgr,mask,K,D)
    image_dir=folder/'img/0';mask_dir=image_dir/'mask';mask_dir.mkdir(parents=True,exist_ok=True)
    rgbpath=image_dir/f'img_{index:04d}.png';maskpath=mask_dir/f'img_{index:04d}.png'
    if not cv2.imwrite(str(rgbpath),rgb) or not cv2.imwrite(str(maskpath),foreground.astype(np.uint8)*255):
        raise IOError('Could not save prepared official input')
    record=dict(camera_id=cid,contiguous_index=index,frame=frame,rgb_key=rgbkey,mask_key=maskkey,
                rgb=str(rgbpath),mask=str(maskpath),native_hw=[height,width],output_hw=[h,w],
                native_K=K.tolist(),native_D=D.tolist(),native_to_input=A.tolist(),
                K=processed_K.tolist(),E=E.tolist(),official_RT_c2w=c2w.tolist(),
                preprocess='undistort at native K, AREA RGB / NEAREST binary mask; exact half-pixel resize K; lossless PNG',
                foreground_pixels=int(foreground.sum()))
    return record,foreground


def nhr_cameras(folder,records):
    """按原NHR的z/x/y/t列布局写c2w，索引仅含本split。"""
    rows=[];intrinsics=[]
    for record in records:
        c2w=np.asarray(record['official_RT_c2w'])
        rows.append(np.concatenate([c2w[:3,2],c2w[:3,0],c2w[:3,1],c2w[:3,3]]))
        intrinsics.append('\n'.join(' '.join(f'{v:.12g}' for v in row) for row in record['K']))
    np.savetxt(folder/'CamPose.inf',np.asarray(rows),fmt='%.12g')
    (folder/'Intrinsic.inf').write_text('\n\n'.join(intrinsics)+'\n',encoding='ascii')


def box_coverage(record,mask,bounds):
    """记录先验box投影外的训练mask，不以GT扩box或删mask。"""
    corners=np.array(list(itertools.product(*zip(bounds['min'],bounds['max']))))
    E,K=np.asarray(record['E']),np.asarray(record['K'])
    camera=np.einsum('ni,ji->nj',corners,E[:3,:3],optimize=False)+E[:3,3]
    uvw=np.einsum('ni,ji->nj',camera,K,optimize=False);uv=uvw[:,:2]/uvw[:,2:3]
    yy,xx=np.where(mask)
    inside=(xx>=uv[:,0].min())&(xx<=uv[:,0].max())&(yy>=uv[:,1].min())&(yy<=uv[:,1].max())
    return dict(camera_id=record['camera_id'],foreground_count=len(xx),outside_projected_box_rectangle=int((~inside).sum()),
                all_box_corners_positive_Z=bool((camera[:,2]>0).all()),
                scope='2D enclosing rectangle is a necessary coverage check, not proof of complete 3D clothing containment')


def prepare(sequence,frame):
    """冻结60实际相机的52训练与8留出，源拟合先验仅提供bbox。"""
    sample=f'DNA_RENDERING/{sequence[:4]}/{sequence}/{frame:04d}'
    folder=OUT/'teacher'/f'{sequence}_{frame:04d}';train_dir=folder/'train';held_dir=folder/'heldout'
    train_dir.mkdir(parents=True,exist_ok=True);held_dir.mkdir(exist_ok=True)
    main_path=DATA/'dna_rendering_part1_main'/f'{sequence}.smc'
    annot_path=DATA/'dna_rendering_part1_annotations'/f'{sequence}_annots.smc'
    prior=ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021/inputs'/sample.replace('/','__')/'SOURCE_FITTED/INPUTS.npz'
    with np.load(prior) as z:
        center=z['unit_center'].reshape(3);scale=float(z['unit_scale'].reshape(-1)[0]);source_E=z['E'][0]
    bounds=dict(min=(center-scale/2).tolist(),max=(center+scale/2).tolist(),
                prior=str(prior),role='existing six-source RGB fitted prior cube; bbox only, not shape supervision',
                excluded_teacher_heldout=HELD,units='metres',Kinect_or_scan_consumed=False)
    dump(folder/'BBOX.json',bounds)
    train_ids=[cid for cid in range(60) if cid not in HELD]
    dump(folder/'SPLIT.json',dict(sample=sample,train=train_ids,heldout=HELD,
          rule='Frozen before teacher fitting; exclude original student six source and source-fit camera17 from heldout',
          student_source=[2,11,20,26,35,41],source_fit=[2,11,17,26,35,41],teacher_images_used_for_bbox=False))
    records=[];held=[];coverage=[]
    with h5py.File(main_path,'r') as main,h5py.File(annot_path,'r') as annots:
        # 第一个原学生相机闭合物理世界，失败则不启动教师。
        E2=inverse_rt(np.asarray(annots['Camera_Parameter/02/RT'],float))
        delta=float(np.max(np.abs(E2-source_E[0])))
        if delta>1e-5:raise ValueError(f'Official camera and source-fit world differ: max {delta}')
        for cid in range(60):
            selected=cid not in HELD;target=train_dir if selected else held_dir
            collection=records if selected else held
            record,mask=export_view(main,annots,cid,frame,target,len(collection));collection.append(record)
            if selected:coverage.append(box_coverage(record,mask,bounds))
            print(f'EXPORTED camera={cid} split={"train" if selected else "heldout"}',flush=True)
    nhr_cameras(train_dir,records);nhr_cameras(held_dir,held)
    centers=np.asarray([np.asarray(r['official_RT_c2w'])[:3,3] for r in records]);pairs=[]
    for index,record in enumerate(records):
        distances=np.linalg.norm(centers-centers[index],axis=1);distances[index]=np.inf
        neighbors=np.argsort(distances,kind='stable')[:4].tolist()
        pairs.append(dict(reference_index=index,reference_camera=record['camera_id'],neighbor_indices=neighbors,
                          neighbor_cameras=[records[j]['camera_id'] for j in neighbors]))
    pairfile=folder/'pairs.txt';pairfile.write_text('\n'.join(' '.join(map(str,[p['reference_index']]+p['neighbor_indices'])) for p in pairs)+'\n',encoding='ascii')
    dump(folder/'PAIRS.json',dict(rule='Four closest actual calibrated camera centers among the 52 training cameras; no image score or heldout',pairs=pairs))
    for sub in ['visual_hull','ncc']: (folder/sub).mkdir(exist_ok=True)
    common=f'data_path = "{train_dir.as_posix()}"\ndata_type = nhr\nnum = 52\n'
    (folder/'space_carving.conf').write_text(common+f'bbox_file = "{(folder/"BBOX.json").as_posix()}"\nout_mesh_dire = "{(folder/"visual_hull").as_posix()}"\n',encoding='utf-8')
    conf=common+'w=1224\nh=1024\nsap_res=512\nsig=4\nnum_points=30000\nnum_sample=50000\nh_patch_size=5\nncc_thresh=0.5\nlr=0.0001\natol=0.01\nncc_weight=2\nmask_weight=10\nrgb_ncc=False\nsparse_file=none\nneed_mask=True\n'
    conf+=f'pair_file="{pairfile.as_posix()}"\ninput_mesh_dire="{(folder/"visual_hull").as_posix()}"\nout_mesh_dire="{(folder/"ncc").as_posix()}"\n'
    (folder/'ncc.conf').write_text(conf,encoding='utf-8')
    dump(folder/'INPUT_RECEIPT.json',dict(sample=sample,main=str(main_path),annotations=str(annot_path),train=records,heldout=held,
        source_fit_camera2_E_max_abs=delta,bbox=bounds,bbox_mask_coverage=coverage,
        coordinate='Official calibrated metres; identity world transform',teacher_forward_or_optimizer_updates=0,
        heldout_policy='No heldout image/mask enters VH, NCC, neighbor selection, or source pose fitting'))
    print(str(folder/'INPUT_RECEIPT.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',default='0097_04');parser.add_argument('--frame',type=int,default=0)
    args=parser.parse_args();prepare(args.sequence,args.frame)
