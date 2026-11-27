"""Prepare frozen DNA/TH development inference inputs without geometry targets.

Only named RGB/mask/calibration/SMPL-X members are opened from the old source
archives. No feature packet, scan, Kinect depth, or target member is consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, DATASETS, REPO as ROOT

BASE = ARTIFACTS/'stage47R_r25_production_cpu_smoke/body_stabilize/strict_fusion_clothing_exit_audit_20261003'
STUDY = BASE / 'U800_GEOMETRY_ASSOCIATION_20261013'
BINDINGS = BASE / 'LONG_TERM_VGGT_20261006/AUTHORITATIVE_RGB_INPUTS/CACHE_BINDINGS.json'
SOURCE_BINDINGS = BASE / 'LONG_TERM_VGGT_20261006/REFERENCE_CONTROL/REFERENCE_SOURCE_BINDINGS.json'
DEVELOPMENT = STUDY / 'DEVELOPMENT_MANIFEST.json'
COORD = STUDY / 'COORDINATE_COMPATIBLE_SUPERVISION_20261015'
SOURCE_IDS = [2, 11, 17, 26, 35, 41]
PHOTO_IDS = [23, 44]
HARD = ['DNA_RENDERING/0097/0097_04/0000', 'DNA_RENDERING/0019/0019_06/0224']
INPUT_KEYS = ['rgb','mask','K','R','t','smplx_xyz','smplx_faces','smplx_xyz_canonical','smplx_normals_canonical']


def read_json(path):
    """只读明确登记的来源元数据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """在本轮独立目录保存可检查记录。"""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def source_calibrated_world(rotation, translation, vertices):
    """保持TRAIN4标定世界；相机RMS仅记录，不额外归一化输入。"""
    rotation, translation, vertices = [np.asarray(x,np.float64) for x in (rotation,translation,vertices)]
    centers = -np.einsum('vji,vj->vi',rotation,translation)
    scale = np.sqrt(np.mean(np.sum((centers-centers.mean(0))**2,axis=1)))
    E = np.broadcast_to(np.eye(4),(6,4,4)).copy()
    E[:,:3,:3],E[:,:3,3] = rotation,translation
    return E,vertices,np.eye(3),np.zeros(3),float(scale)


def orbit_cameras(K,E,center,up):
    """仅以输入相机半径、人体中心和竖直轴构造固定36视点。"""
    up = up/np.linalg.norm(up)
    start = -E[0,:3,:3].T @ E[0,:3,3]-center
    cameras=[]
    for yaw in range(0,360,10):
        angle=np.deg2rad(yaw)
        offset=start*np.cos(angle)+np.cross(up,start)*np.sin(angle)+up*np.dot(up,start)*(1-np.cos(angle))
        eye=center+offset; forward=center-eye; forward/=np.linalg.norm(forward)
        right=np.cross(forward,up); right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        R=np.stack((right,down,forward)); new=np.eye(4); new[:3,:3]=R;new[:3,3]=-R@eye
        cameras.append(dict(view_id=f'orbit_{yaw:03d}',yaw=yaw,role='synthetic_orbit_camera',K=K[0].tolist(),E=new.tolist()))
    return cameras


def restore_fixed_sources(sid,values,lineage,data_root,restored_main_dir=None):
    """完整官方容器恢复后，仅解码指定同帧六源RGB、mask与标定。"""
    import h5py
    seq=sid.split('/')[2];frame=int(sid.split('/')[-1])
    main=(Path(restored_main_dir) if restored_main_dir else Path(data_root)/'dna_rendering_part1_main')/f'{seq}.smc'
    annot=Path(data_root)/'dna_rendering_part1_annotations'/f'{seq}_annots.smc'
    images=[];masks=[];Ks=[];Rs=[];ts=[];records=[]
    with h5py.File(main,'r') as fm,h5py.File(annot,'r') as fa:
        for cid in SOURCE_IDS:
            rgb_key=f'Camera_5mp/{cid}/color/{frame}';mask_key=f'Mask/{cid}/mask/{frame}'
            bgr=cv2.imdecode(np.asarray(fm[rgb_key]),cv2.IMREAD_COLOR)
            mask=cv2.imdecode(np.asarray(fa[mask_key]),cv2.IMREAD_UNCHANGED)
            cal=fa[f'Camera_Parameter/{cid:02d}'];K=np.asarray(cal['K'],float);D=np.asarray(cal['D'],float)
            E=np.linalg.inv(np.asarray(cal['RT'],float));h,w=bgr.shape[:2]
            sx,sy=433/w,518/h;left=int(round(259-K[0,2]*sx))
            A=np.asarray([[sx,0,left+.5*(sx-1)],[0,sy,.5*(sy-1)],[0,0,1]])
            mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
            rgb=cv2.cvtColor(cv2.resize(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR),(433,518),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            if mask.ndim==3: mask=mask.max(2)
            mask=cv2.resize(cv2.remap(mask,mx,my,cv2.INTER_NEAREST),(433,518),interpolation=cv2.INTER_NEAREST)>0
            image=np.zeros((3,518,518),np.float32);image[:,:,left:left+433]=np.moveaxis(rgb,2,0).astype(np.float32)/255
            foreground=np.zeros((1,518,518),bool);foreground[:,:,left:left+433]=mask[None]
            images.append(image);masks.append(foreground);Ks.append(A@K);Rs.append(E[:3,:3]);ts.append(E[:3,3])
            records.append(dict(camera_id=cid,frame=frame,rgb=f'{main}::{rgb_key}',mask=f'{annot}::{mask_key}',
                                calibration=f'{annot}::Camera_Parameter/{cid:02d}',native_hw=[h,w],native_to_input=A.tolist()))
    values.update(rgb=np.stack(images),mask=np.stack(masks),K=np.asarray(Ks,np.float32),R=np.asarray(Rs,np.float32),t=np.asarray(ts,np.float32))
    lineage.update(historical_original_source_camera_ids=lineage['native_rgb_camera_ids'],native_rgb_camera_ids=[f'{v:02d}' for v in SOURCE_IDS],
                   fixed_source_restoration=records,restoration_reads='RGB/mask/K/D/RT only; no scan/Kinect/SMPL annotation replaced')
    return values,lineage


def photo_cameras(sid,folder,R0,t0,scale):
    """真实23/44相机；评价照片独立保存且不进入模型输入。"""
    seq=sid.split('/')[2]; frame=int(sid.split('/')[-1])
    native=BASE/'MENTOR_CALIBRATED_RGB_TEACHER_20261012/NATIVE'/seq
    receipt_path=native/'NATIVE_RECEIPT.json'
    if not receipt_path.exists():
        return [],[],dict(status='unavailable_local_native_camera_and_photos',requested_view_ids=PHOTO_IDS)
    receipt=read_json(receipt_path)
    if receipt['frame'] != frame:
        return [],[],dict(status='native_receipt_other_frame',requested_view_ids=PHOTO_IDS,available_frame=receipt['frame'])
    indexed={r['camera']:r for r in receipt['views']}
    cameras=[];photos=[]
    for cid in PHOTO_IDS:
        row=indexed[cid]; cal=row['calibration']; K=np.asarray(cal['K'],float); D=np.asarray(cal['D'],float)
        raw_E=np.linalg.inv(np.asarray(cal['RT'],float))
        h,w=row['native_hw']; sx,sy=433/w,518/h
        left=int(round(259-K[0,2]*sx))
        A=np.array([[sx,0,left+.5*(sx-1)],[0,sy,.5*(sy-1)],[0,0,1]])
        virtual_K=A@K
        E=np.eye(4); E[:3,:3]=raw_E[:3,:3]@R0.T
        E[:3,3]=(raw_E[:3,3]-E[:3,:3]@t0)/scale
        cameras.append(dict(view_id=cid,role='historically_exposed_non_source_photo_development',K=virtual_K.tolist(),E=E.tolist()))
        photo_folder=folder/'EVAL_ONLY';photo_folder.mkdir(exist_ok=True)
        rgb=cv2.imread(str(native/f'cam{cid:02d}_RGB.png'))
        mask=cv2.imread(str(native/f'cam{cid:02d}_ANNOTATION.png'),cv2.IMREAD_UNCHANGED)
        mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
        rgb=cv2.resize(cv2.remap(rgb,mx,my,cv2.INTER_LINEAR),(433,518),interpolation=cv2.INTER_AREA)
        if mask.ndim==3: mask=mask.max(2)
        mask=cv2.resize(cv2.remap(mask,mx,my,cv2.INTER_NEAREST),(433,518),interpolation=cv2.INTER_NEAREST)>0
        canvas=np.zeros((518,518,3),np.uint8);canvas[:,left:left+433]=rgb
        canvas_mask=np.zeros((518,518),np.uint8);canvas_mask[:,left:left+433]=mask.astype(np.uint8)*255
        rgb_path=photo_folder/f'cam{cid:02d}_RGB.png';mask_path=photo_folder/f'cam{cid:02d}_MASK.png'
        cv2.imwrite(str(rgb_path),canvas);cv2.imwrite(str(mask_path),canvas_mask)
        photos.append(dict(query_index=len(cameras)-1,view_id=cid,rgb_path=str(rgb_path.resolve()),mask_path=str(mask_path.resolve()),
                           source_rgb=str(native/f'cam{cid:02d}_RGB.png'),source_mask=str(native/f'cam{cid:02d}_ANNOTATION.png'),
                           native_to_input=A.tolist(),left=left,role='evaluation_only_never_model_input'))
    return cameras,photos,dict(status='available',receipt=str(receipt_path),requested_view_ids=PHOTO_IDS,
                               processing='undistort K/D then AREA RGB, NEAREST mask 433x518 and principal-centered P1; halfpixel K')


def prepare_real_query_photos(output,data_root,restored_main_dir):
    """独立补充固定23/44照片与相机，不读写模型输入或已运行查询。"""
    import h5py
    output=Path(output).resolve();manifest=read_json(output/'MANIFEST.json');records=[]
    for item in manifest['samples']:
        sid=item['id']
        if not sid.startswith('DNA_RENDERING/') or sid in HARD: continue
        _,subject,seq,frame_text=sid.split('/');frame=int(frame_text);folder=output/sid.replace('/','__')
        if (folder/'EVAL_REAL_23_44.json').exists():
            records.append(dict(id=sid,status='actual_photos_decoded',photo_count=2,
                query_cameras=str(folder/'ADDITIONAL_QUERY_CAMERAS.json'),eval_photos=str(folder/'EVAL_REAL_23_44.json')))
            continue
        main=(Path(restored_main_dir) if restored_main_dir else Path(data_root)/'dna_rendering_part1_main')/f'{seq}.smc'
        if not main.exists():
            main=Path(data_root)/'dna_rendering_part1_main'/f'{seq}.smc'
        if not main.exists():
            records.append(dict(id=sid,status='missing_complete_main',main=str(main),fixed_query_ids=PHOTO_IDS));continue
        annot=Path(data_root)/'dna_rendering_part1_annotations'/f'{seq}_annots.smc'
        photos=folder/'EVAL_REAL_23_44';photos.mkdir(exist_ok=True)
        cameras=[];rows=[]
        with h5py.File(main,'r') as fm,h5py.File(annot,'r') as fa:
            for index,cid in enumerate(PHOTO_IDS):
                rgb_key=f'Camera_5mp/{cid}/color/{frame}';mask_key=f'Mask/{cid}/mask/{frame}'
                bgr=cv2.imdecode(np.asarray(fm[rgb_key]),cv2.IMREAD_COLOR)
                mask=cv2.imdecode(np.asarray(fa[mask_key]),cv2.IMREAD_UNCHANGED)
                cal=fa[f'Camera_Parameter/{cid:02d}'];K=np.asarray(cal['K'],float);D=np.asarray(cal['D'],float)
                RT=np.asarray(cal['RT'],float);E=np.linalg.inv(RT);h,w=bgr.shape[:2]
                sx,sy=433/w,518/h;left=int(round(259-K[0,2]*sx))
                A=np.asarray([[sx,0,left+.5*(sx-1)],[0,sy,.5*(sy-1)],[0,0,1]])
                mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
                rgb=cv2.resize(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR),(433,518),interpolation=cv2.INTER_AREA)
                if mask.ndim==3: mask=mask.max(2)
                fg=cv2.resize(cv2.remap(mask,mx,my,cv2.INTER_NEAREST),(433,518),interpolation=cv2.INTER_NEAREST)>0
                canvas=np.zeros((518,518,3),np.uint8);canvas[:,left:left+433]=rgb
                canvas_mask=np.zeros((518,518),np.uint8);canvas_mask[:,left:left+433]=fg.astype(np.uint8)*255
                rgb_path=photos/f'cam{cid:02d}_RGB.png';mask_path=photos/f'cam{cid:02d}_MASK.png'
                cv2.imwrite(str(rgb_path),canvas);cv2.imwrite(str(mask_path),canvas_mask)
                virtual_K=A@K
                camera=dict(view_id=cid,query_index=36+index,role='fixed_real_non_source_historical_development',K=virtual_K.tolist(),E=E.tolist())
                cameras.append(camera)
                rows.append(dict(**camera,rgb_path=str(rgb_path),mask_path=str(mask_path),
                    source_rgb=f'{main}::{rgb_key}',source_mask=f'{annot}::{mask_key}',calibration=f'{annot}::Camera_Parameter/{cid:02d}',
                    native_hw=[h,w],native_K=K.tolist(),native_D=D.tolist(),native_RT=RT.tolist(),native_to_input=A.tolist(),left=left,
                    units='source calibrated world metres, no world transform',mask_pixels=int(fg.sum()),evaluation_only=True,
                    processing='undistort K/D, RGB AREA and mask NEAREST to433x518, P1 with matching halfpixel K'))
        write_json(folder/'ADDITIONAL_QUERY_CAMERAS.json',dict(cameras=cameras,append_only=True,original_queries_unchanged=True,model_reads='K/E only'))
        write_json(folder/'EVAL_REAL_23_44.json',dict(id=sid,rows=rows,fixed_query_ids=PHOTO_IDS,status='actual_photos_decoded',
                   evaluation_only=True,input_npz_unchanged=True,selection='user-fixed23/44 before result; no output used'))
        records.append(dict(id=sid,status='actual_photos_decoded',photo_count=len(rows),
                       query_cameras=str(folder/'ADDITIONAL_QUERY_CAMERAS.json'),eval_photos=str(folder/'EVAL_REAL_23_44.json')))
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    write_json(output/'EVAL_REAL_23_44_MANIFEST.json',dict(samples=records,fixed_query_ids=PHOTO_IDS,input_npz_unchanged=True,
               original_query_files_unchanged=True,model_or_GPU_calls=0))


def prepare(output, hard_only=False, restore=False, data_root=(DATASETS/'dna_rendering'),restored_main_dir=None):
    """冻结18例清单，准备合法输入并单列缺失固定源的样本。"""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if output == (ROOT/'results/DNA_TRANSFER_U2000/INPUTS').resolve():
        raise ValueError('旧INPUTS保留为REFERENCE_GAUGE_DIAGNOSTIC；请选择INPUTS_CALIBRATED_WORLD')
    binding=read_json(BINDINGS); development=read_json(DEVELOPMENT)
    source_records={r['sample_id']:r for r in read_json(SOURCE_BINDINGS)['fixed16_samples']}
    all_ids=HARD+development['dev_ids']
    ready={r['id']:r for r in read_json(output/'MANIFEST.json')['samples']} if restore and (output/'MANIFEST.json').exists() else {}
    # Freeze membership and camera roles before running any model.
    contract=dict(samples=all_ids,source_view_ids=SOURCE_IDS,photo_evaluation_view_ids=PHOTO_IDS,
                  DNA_development_frames=8,DNA_development_identities=['0025','0034','0095'],
                  historical_development_not_blind=True,synthetic_orbit_yaws=list(range(0,360,10)),
                  source_manifest=str(DEVELOPMENT),training=False)
    write_json(output/'FIXED_PROTOCOL.json',contract)
    samples=[];pending=[]
    for sid in (HARD if hard_only else all_ids):
        if sid in ready:
            # 恢复模式只消费pending，不重写正在推理的15例任何输入文件。
            samples.append(ready[sid]);continue
        slug=sid.replace('/','__');folder=output/slug;folder.mkdir(exist_ok=True)
        source=Path(binding['source_paths'][sid])
        with np.load(source,allow_pickle=False) as z:
            values={k:z[k].copy() for k in INPUT_KEYS}
            lineage=json.loads(str(z['source_lineage_json'])) if 'source_lineage_json' in z else {}
        if not lineage and sid in source_records:
            original=Path(source_records[sid]['original_source_absolute_path'])
            with np.load(original,allow_pickle=False) as z:
                lineage=json.loads(str(z['source_lineage_json']))
            lineage['metadata_source_original_archive']=str(original)
            lineage['reference_copy_record']=source_records[sid]
        dna=sid.startswith('DNA_RENDERING/')
        restored=False
        if restore and dna and [int(v) for v in lineage['native_rgb_camera_ids']]!=SOURCE_IDS:
            main=(Path(restored_main_dir) if restored_main_dir else Path(data_root)/'dna_rendering_part1_main')/(sid.split('/')[2]+'.smc')
            if main.exists():
                values,lineage=restore_fixed_sources(sid,values,lineage,data_root,restored_main_dir);restored=True
        if dna and lineage and [int(v) for v in lineage['native_rgb_camera_ids']]!=SOURCE_IDS:
            record=dict(id=sid,status='pending_missing_fixed_source',required_source_view_ids=SOURCE_IDS,
                        actual_cached_source_view_ids=[int(v) for v in lineage['native_rgb_camera_ids']],
                        missing_source_view_ids=sorted(set(SOURCE_IDS)-set(int(v) for v in lineage['native_rgb_camera_ids'])),
                        original_metadata=lineage,existing_draft_input_not_authorized=True,
                        reason='Historical source substitution differs from transfer protocol; do not run this draft input')
            pending.append(record);write_json(folder/'PENDING_FIXED_SOURCE.json',record)
            continue
        K=values['K'].copy()
        # Existing camera-only member of corrected-coordinate products. Targets are never read.
        corrected=COORD/'INPUTS'/slug/'KINECT_COORDINATE_TARGET.npz'
        if dna and not restored:
            with np.load(corrected,allow_pickle=False) as z: K=z['K'].copy()
        E,xyz,R0,t0,scale=source_calibrated_world(values['R'],values['t'],values['smplx_xyz'])
        center=(xyz.min(0)+xyz.max(0))*.5;cube=float(np.ptp(xyz,axis=0).max()*1.3)
        inputs=dict(images=values['rgb'][None],masks=values['mask'][:,0][None],K=K[None],E=E[None].astype(np.float32),
                    smpl_vertices=xyz[None].astype(np.float32),smpl_faces=values['smplx_faces'],
                    smpl_canonical=values['smplx_xyz_canonical'][None],smpl_normals=values['smplx_normals_canonical'][None],
                    unit_center=center[None].astype(np.float32),unit_scale=np.asarray([cube],np.float32))
        mask_path=BASE/'LOCAL_INPUTS/DNA_RENDERING__0019__0019_06__0224__LEARNED_RGB_INSTANCE_MASK.npz'
        if sid==HARD[1]:
            with np.load(mask_path,allow_pickle=False) as z:
                masks=z['foreground'] if 'foreground' in z else z['mask']
                inputs['masks']=(masks[:,0] if masks.ndim==4 else masks)[None].astype(bool)
        cameras,photos,photo_record=photo_cameras(sid,folder,R0,t0,1.0) if dna else ([],[],dict(status='TH_synthetic_orbit_only_no_real_query_photos_in_source_cache'))
        up=R0@np.asarray([0,-1,0] if dna else [0,0,1],float)
        cameras+=orbit_cameras(K,E,center,up)
        # Prepared batches are immutable while runtime may be consuming them.
        if restored or not (folder/'INPUTS.npz').exists():
            np.savez_compressed(folder/'INPUTS.npz',**inputs)
        write_json(folder/'QUERY_CAMERAS.json',dict(cameras=cameras,model_reads='K/E only; RGB/mask/depth excluded'))
        write_json(folder/'EVAL_PHOTOS.json',dict(rows=photos,availability=photo_record))
        normalization=dict(R0=R0.tolist(),t0=t0.tolist(),camera_center_RMS_source_units=scale,
            formula='X_input=X_source; E_input=[source_R,source_t]; world_scale_factor=1; camera_RMS computed internally by network for depth only',
            units='metres in inherited calibrated source producer contract; no new unit conversion',
            cube_center=center.tolist(),cube_side=cube,cube_rule='1.3 * maximum source SMPL-X bbox extent; isotropic; no scan statistics',
            axis='DNA original calibrated coordinates; no THuman scan-axis transform' if dna else 'TH existing calibrated source world; no scan read')
        pose_condition='given_pose_controlled_experiment; source-only provenance unproved'
        if not dna and lineage.get('accepted_prior_geometry'):
            producer=Path(lineage['accepted_prior_geometry']).parent.parent/'A/producer_manifest.json'
            fit=read_json(producer);rows=fit.get('source_rows',[])
            lineage['smplx_fit_producer']=str(producer)
            lineage['smplx_fit_source_views']=[r.get('view_id') for r in rows]
            lineage['smplx_fit_source_RGBs']=[r.get('rgb_path') for r in rows]
            if len(rows)==6 and {r.get('rgb_path') for r in rows}==set(lineage.get('rgb_paths',[])):
                pose_condition='documented_six_source_RGB_keypoints_known_camera_fit; no_scan_fit'
        lineage.update(input_source=str(source),input_member_allowlist=INPUT_KEYS,source_view_ids=SOURCE_IDS if dna else lineage['native_rgb_camera_ids'],
            pose_condition=pose_condition,world_contract='TRAIN4_source_calibrated_world_preserved',
            world_up=[0,-1,0] if dna else [0,0,1],units='metres; inherited calibrated source producer contract; scale factor1',
            scan_to_model_transform=None if dna else dict(matrix=[[1,0,0],[0,0,-1],[0,1,0]],translation=[0,0,0],scale=1,
                formula='TH raw scan xyz -> (x,-z,y); evaluation only, source world unchanged',role='evaluation_only_not_input'),
            units_evidence=dict(source_contract=str(ROOT/'local_r20_tools/build_r20_dna_source_only.py') if dna else str(ROOT/'research/mentor_human_model/data/prepare_train4.py'),
                world_implementation='source R/t and smplx_xyz copied unchanged; no numeric unit conversion'),
            smplx_source=lineage.get('smplx_source','existing fixed16 source SMPL-X; source-only fitting provenance unproved'),
            no_RGB_only_reconstruction_claim=True,normalization=normalization,K_source='restored actual RGB pixel-center K from official calibration' if restored else (str(corrected) if dna else str(source)),
            RGB_features='must encode this exact current six-image batch in this inference; no THuman feature reuse',
            mask_source=str(mask_path) if sid==HARD[1] else str(source),evaluation_photos=photo_record,
            input_forbidden_members_not_read=['target_depth','target_depth_valid','target_normal','target_normal_valid','scan','Kinect'],
            historical_development_not_blind=True)
        write_json(folder/'LINEAGE.json',lineage)
        samples.append(dict(id=sid,status='ready',role='hard_diagnostic' if sid in HARD else 'historical_development',
                            input_npz=f'{slug}/INPUTS.npz',query_cameras=f'{slug}/QUERY_CAMERAS.json',
                            eval_photos=f'{slug}/EVAL_PHOTOS.json',lineage=lineage))
        write_json(output/'MANIFEST.json',dict(protocol='U2000_transfer_calibrated_source_world',samples=samples,pending_samples=pending,
                   fixed_protocol='FIXED_PROTOCOL.json',prepared_count=len(samples),planned_count=18))
        print(json.dumps(dict(prepared=sid,input=str(folder/'INPUTS.npz'),query_count=len(cameras),photo_count=len(photos)),ensure_ascii=False),flush=True)
    write_json(output/'MANIFEST.json',dict(protocol='U2000_transfer_calibrated_source_world',samples=samples,pending_samples=pending,
               fixed_protocol='FIXED_PROTOCOL.json',prepared_count=len(samples),planned_count=18))


def main():
    """CPU数据入口，不运行训练、网络或GPU。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/DNA_TRANSFER_U2000/INPUTS_CALIBRATED_WORLD')
    parser.add_argument('--hard-only',action='store_true')
    parser.add_argument('--restore-fixed-sources',action='store_true',help='完整官方SMC就绪后恢复3帧缺失六源；不接受partial')
    parser.add_argument('--dna-root',type=Path,default=(DATASETS/'dna_rendering'))
    parser.add_argument('--restored-main-dir',type=Path,help='独立恢复的完整SMC目录，不改原DriveFS数据')
    parser.add_argument('--eval-real-queries-only',action='store_true',help='只补评价照片与追加相机，不读写模型输入')
    args=parser.parse_args()
    if args.eval_real_queries_only:
        prepare_real_query_photos(args.output,args.dna_root,args.restored_main_dir)
    else:
        prepare(args.output,args.hard_only,args.restore_fixed_sources,args.dna_root,args.restored_main_dir)


if __name__=='__main__':
    main()
