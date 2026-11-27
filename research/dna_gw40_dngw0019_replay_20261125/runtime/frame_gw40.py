"""Run a candidate frame under the archived 0019 GW40 camera and optimizer contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, OFFICIAL as HIST, REPO as ROOT, RESULTS, VENDOR

RESULT = RESULTS/'DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125'
CASES = {'0008': ('0008_01','READ48_RECEIPT.json'), '0047': ('0047_01','RECOVERED_READ48_RECEIPT.json')}

def read(path):
    return path.read_text(encoding='utf-8-sig')

def document(path):
    return json.loads(read(path))

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def now():
    return datetime.now().astimezone().isoformat()

def inputs(code, run):
    import numpy as np
    from PIL import Image
    seq,receipt = CASES[code]
    sid=f'DNA_RENDERING/{code}/{seq}/0000'
    base=ARTIFACTS/'DNA_MULTISUBJECT_REPAIR_20261024/inputs'/sid.replace('/','__')
    orig=document(base/receipt)
    roles=document(base/'SPLIT40_8.json')
    train=roles['train']
    held=roles['heldout_sorted']
    assert orig['sample_id']==roles['sample_id']==sid
    assert len(train)==40 and len(held)==8 and not set(train)&set(held) and sorted(train+held)==list(range(48))
    rows=[]
    for r in sorted(orig['records'],key=lambda x:x['camera_id']):
        cid=r['camera_id']
        assert r['frame']==0 and cid==len(rows) and r['output_hw']==[1024,1224]
        width,height=1224,1024
        with Image.open(r['rgb']) as rgb, Image.open(r['mask']) as mask:
            assert rgb.size==mask.size==(width,height)
            foreground=int(np.count_nonzero(np.asarray(mask.convert('L'))>=128))
        rows.append(dict(r,subset='train' if cid in train else 'held',rgb_header=dict(wh=[width,height]),mask_pixels=foreground))
    assert len(rows)==48 and set([2,26,11,35])<=set(train)
    contract=dict(sample=sid,train_ids=train,held_ids=held,rows=rows,role_receipt=str(base/'SPLIT40_8.json'),
                  source_receipt=str(base/receipt),source_main=orig['main_archive'],source_annotation=orig['annotation_archive'],
                  four_body_mask_pixels={str(i):rows[i]['mask_pixels'] for i in [2,26,11,35]},mask_handling='Same-frame acquired foreground mask. No 0019 cam09 manual polygon transferred.')
    save(run/'FRAME_INPUT.json',contract)
    return contract

def prepare(code,run):
    import numpy as np
    import pycolmap
    sys.path.insert(0,str(ROOT))
    from research.dna_gw_second_identity_0102_20261123 import build34
    contract=inputs(code,run)
    rows=[r for r in contract['rows'] if r['camera_id'] in contract['train_ids']]
    build34.SIFT=run/'known_pose_train40'
    build34.RUN=run
    build34.sift(rows)
    original=pycolmap.Reconstruction(str(build34.SIFT/'known_model'))
    sparse=pycolmap.Reconstruction(str(build34.SIFT/'sparse/0'))
    assert sparse.num_reg_images()==40 and sparse.num_points3D()>0
    dK=max(float(np.max(np.abs(sparse.camera(i).params-original.camera(i).params))) for i in original.cameras)
    dE=max(float(np.max(np.abs(sparse.image(i).cam_from_world().matrix()-original.image(i).cam_from_world().matrix()))) for i in original.images)
    assert dK<1e-8 and dE<1e-6,(dK,dE)
    save(run/'KNOWN_POSE_SIFT_CONTRACT.json',dict(source_index=str(run/'FRAME_INPUT.json'),registered_train=40,held_used=0,SIFT_points=sparse.num_points3D(),K_max_delta=dK,E_max_delta=dE,commands=document(run/'SIFT_CALLS.json')))
    from center_official_input import center

    center(run)

def teacher(run):
    src=run/'official_input_train40'
    model=run/'GW_OFFICIAL_DIRECT_DEFAULT'
    historical=document(HIST/'RUN_CONTRACT.json')
    cmd=list(historical['command'])
    vendor = VENDOR/'gaussian_wrapping'
    cmd[0] = sys.executable
    cmd[2] = str(vendor/'scripts/train_and_extract_gw_ours.py')
    cmd[cmd.index('-s')+1]=str(src)
    cmd[cmd.index('-m')+1]=str(model)
    assert not model.exists() and len(list((src/'images').glob('*.png')))==40
    assert document(run/'CENTERED_INPUT_PROJECTION_RECEIPT.json')['world_points']>0
    launch=dict(start=now(),command=cmd,historical_vendor=str(vendor),historical_run_contract=str(HIST/'RUN_CONTRACT.json'),status='RUNNING',camera_neighbor_implementation='original 11/02 vendor; historic global cap',student_optimizer=0)
    save(run/'GW40_RUN.json',launch)
    with (run/'GW40_OFFICIAL.log').open('wb') as log:
        res=subprocess.run(cmd, cwd=vendor, env=dict(os.environ,TORCH_EXTENSIONS_DIR=str(HIST/'native_jit'),PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
    final=model/'mesh_ours_2pivots_post_texture_refined_999.ply'
    launch.update(end=now(),returncode=res.returncode,status='COMPLETED' if res.returncode==0 and final.is_file() else 'FAILED',texture_final=str(final))
    save(run/'GW40_RUN.json',launch)
    if launch['status']!='COMPLETED': raise RuntimeError(f'Historical GW wrapper status {launch}')

def body(run):
    import cv2
    import numpy as np
    import trimesh
    from PIL import Image
    import body_only as old

    filename = Path(old.__file__)
    contract=document(run/'FRAME_INPUT.json')
    rows=[next(r for r in contract['rows'] if r['camera_id']==i) for i in (2,26,11,35)]
    for r in rows:
        mask=np.asarray(Image.open(r['mask']).convert('L'))>=128
        r['_mask']=cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8))>0
    source=run/'GW_OFFICIAL_DIRECT_DEFAULT/mesh_ours_2pivots_post_texture_refined_999.ply'
    mesh=trimesh.load(source,process=False,maintain_order=True)
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces); colors=np.asarray(mesh.visual.vertex_colors)[:,:3]
    keep=old.supported(v,rows)[f].all(axis=1)
    for a in range(0,len(f),100000):
        g=f[a:a+100000];tri=v[g];valid=keep[a:a+len(g)]
        for p in (tri.mean(1),(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2):
            valid &= old.supported(p,rows)
    faceids=np.flatnonzero(keep);index=np.unique(f[keep]);mapping=np.full(len(v),-1,np.int32);mapping[index]=np.arange(len(index))
    target=run/'human_body_only';target.mkdir(exist_ok=True)
    output=target/'GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    old.save_ply(output,v[index],colors[index],mapping[f[keep]])
    old.save_ply(target/'GW_HUMAN_BODY_ONLY_POINTS.ply',v[index],colors[index])
    for key,val in [('ORIGINAL_VERTEX_INDICES',index),('ORIGINAL_FACE_INDICES',faceids),('REMOVED_FACE_INDICES',np.flatnonzero(~keep))]: np.save(target/(key+'.npy'),val)
    check=trimesh.load(output,process=False,maintain_order=True)
    assert np.array_equal(check.vertices,v[index].astype(np.float32)) and np.array_equal(check.faces,mapping[f[keep]]) and np.array_equal(check.visual.vertex_colors[:,:3],colors[index])
    selected=[{key:r[key] for key in ('camera_id','rgb','mask','K','E')} for r in rows]
    save(target/'BODY_ONLY_RECEIPT.json',dict(source=str(source),mesh=str(output),original_rule=str(filename),source_vertices=len(v),source_faces=len(f),faces=len(faceids),vertices=len(index),removed_faces=int((~keep).sum()),body_mask_ids=[2,26,11,35],selection=selected,threshold=128,dilation_square=17,votes=3,positions=7,held_used=0,optimizer_updates=0,QA='PENDING',original_face_index='ORIGINAL_FACE_INDICES.npy'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','teacher','body'])
    parser.add_argument('code',choices=list(CASES))
    args=parser.parse_args()
    run=RESULT/('GW40_'+args.code);run.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare':prepare(args.code,run)
    if args.action=='teacher':teacher(run)
    if args.action=='body':body(run)
