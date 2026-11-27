"""只读原0019数组，冻结源像素支持与旧观察区域；不调用网络或表面算子。"""
from pathlib import Path
import json
import numpy as np
import trimesh

from research.dna_gw40_dngw0019_replay_20261125.paths import REPO as ROOT, RESULTS

OUT = RESULTS/'DNA_0019_NEUS2_SURFACE_CONSUMPTION_20261028'
OLD=ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021'
CASE='DNA_RENDERING__0019__0019_06__0224'


def capsule(p,chain,radius):
    """原肩肘腕两线段距离，仅是观察通道，不是精确表面标签。"""
    ds=[]
    for a,b in zip(chain[:-1],chain[1:]):
        d=b-a;t=np.clip(((p-a)*d).sum(-1)/(d*d).sum(),0,1)
        ds.append(np.linalg.norm(p-a-t[...,None]*d,axis=-1))
    return np.minimum.reduce(ds)<=radius


def box(p,bounds):
    """复用旧世界窗，保留窗外原始点。"""
    return ((p>=bounds[0])&(p<=bounds[1])).all(-1)


def main():
    """一次CPU实数统计，保存完整布尔支持、配对与来源；不改旧根。"""
    dest=OUT/'CPU_SUPPORT';dest.mkdir(exist_ok=True)
    ip=OLD/'inputs'/CASE/'GIVEN_PRIOR/INPUTS.npz'
    tp=ip.with_name('TARGETS.npz')
    rp=OLD/'evaluation/B_KINECT_U600'/CASE/'GIVEN_PRIOR/source_RAW.npz'
    with np.load(ip) as z:inp={k:z[k] for k in ['images','masks','K','E','unit_center','unit_scale']}
    with np.load(tp) as z:target={k:z[k] for k in z.files}
    with np.load(rp) as z:raw={k:z[k] for k in ['world_points','world_normal','geometric_normal','normal_valid','oriented_points','oriented_normals','mask','unit_center','unit_scale']}
    regionpath=ROOT/'artifacts/thuman20/DNA_0019_ARM_CARVING_DIAG_20261023/0019_来源与运行.json'
    region=json.loads(regionpath.read_text(encoding='utf-8'))
    frozen={k:region[k] for k in ['fixed_arm_worldbox','prior_joint_ids','prior_joint_world_xyz','source_prior_file','prior_channel_radius_m','torso_reference_joint_array_indices','torso_reference_worldbox']}
    frozen.update(source=str(regionpath),role='Original fixed world observation windows and source-fitted shoulder-elbow-wrist chain; not clothing GT; no independent three joint boxes exist')
    (dest/'FROZEN_ORIGINAL_REGIONS.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    F=inp['masks'].astype(bool);V=target['depth_valid'].astype(bool)
    measured_legal=np.isfinite(target['depth'])&(target['depth']>0)
    M=F&V&measured_legal;U=F&~M
    nv=raw['normal_valid'].astype(bool);stride=np.zeros_like(F);stride[:,:,::4,::4]=True
    support=nv&stride
    p=raw['world_points'];n=raw['world_normal'];unit=(p-inp['unit_center'].reshape(1,1,1,1,3))/inp['unit_scale'].reshape(1,1,1,1,1)+.5
    finite=np.isfinite(p).all(-1);out=((unit<=0)|(unit>=1)).any(-1)
    masks={'F':F,'M':M,'U':U,'normal_valid':nv,'native_stride4':support}
    chain=np.array(frozen['prior_joint_world_xyz'])
    regions={'ALL':np.ones_like(F),'OLD_ARM_WORLD_BOX':box(p,np.array(frozen['fixed_arm_worldbox'])),'OLD_SHOULDER_ELBOW_WRIST_CHANNEL':capsule(p,chain,frozen['prior_channel_radius_m']),'OLD_TORSO_REFERENCE_BOX':box(p,np.array(frozen['torso_reference_worldbox']))}
    rows=[]
    for rn,rm in regions.items():
        for sn,sm in masks.items():
            sel=rm&sm
            rows.append(dict(region=rn,support=sn,count=int(sel.sum()),by_source=[int(x.sum()) for x in sel[0]],normal_valid=int((sel&nv).sum()),nonfinite_points=int((sel&~finite).sum()),outside_original_cube=int((sel&out).sum()),nonfinite_learned_normals=int((sel&~np.isfinite(n).all(-1)).sum()),zero_learned_normals=int((sel&(np.linalg.norm(n,axis=-1)<=1e-6)).sum())))
    ids=np.argwhere(support).astype(np.int32)
    pair_p=p[support];pair_n=n[support]
    pair_equal={'points':bool(np.array_equal(pair_p,raw['oriented_points'].reshape(-1,3))),'normals':bool(np.array_equal(pair_n,raw['oriented_normals'].reshape(-1,3)))}
    np.savez_compressed(dest/'FROZEN_SOURCE_PIXEL_SUPPORT.npz',F=F,M=M,U=U,original_target_valid=V,original_target_cameraZ=target['depth'],original_normal_valid=nv,native_stride4_support=support,native_stride4_bvyx=ids,K=inp['K'],E=inp['E'],source_camera_ids=np.array([2,11,17,26,35,41]),unit_center=inp['unit_center'],unit_scale=inp['unit_scale'])
    np.savez_compressed(dest/'U600_NATIVE_STRIDE4_PN_PAIR.npz',source_bvyx=ids,points_world=pair_p,normals_world=pair_n,geometric_normals_world=raw['geometric_normal'][support])
    meshes={}
    for name in ['base_mesh.ply','clothed_mesh.ply']:
        path=rp.parent/name;mesh=trimesh.load(path,process=False,maintain_order=True)
        meshes[name]=dict(path=str(path),vertices=len(mesh.vertices),faces=len(mesh.faces),bounds=mesh.bounds.tolist())
    report=dict(source_input=str(ip),source_target=str(tp),complete_source_RAW=str(rp),original_regions=str(regionpath),definition='F=original INPUTS.masks; M=F & original TARGETS.depth_valid & finite(depth) & depth>0, matching teacher_quality.py original measured; U=F minus M (unknown); native stride4=original normal_valid at ::4,::4. No error selection or nearest-neighbor pairing.',original_raw_mask_equals_F=bool(np.array_equal(raw['mask'].astype(bool),F)),target_valid_outside_F=int((V&~F).sum()),valid_foreground_illegal_depth=int((F&V&~measured_legal).sum()),native_oriented_pair_exact=pair_equal,native_baseline_source='research/mentor_human_model/model/network.py forward: normal_valid ::4 support; world_normal ::4 used by original oriented_normals',rows=rows,meshes=meshes,network_forwards=0,optimizer_updates=0,DPSR_calls=0,GPU_calls=0)
    (dest/'SUPPORT_AND_NATIVE_BASELINE_RECEIPT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'whole_rows':rows[:5],'pair_exact':pair_equal,'meshes':meshes}))


if __name__=='__main__':main()
