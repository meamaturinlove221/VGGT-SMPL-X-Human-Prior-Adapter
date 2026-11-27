"""读取实际源缓存，保持 B 标定世界与 Kinect 原像素支持。"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, external_path

OUT = ARTIFACTS/'DNA_INTEGRATED_REPAIR_20261021'
RAW = external_path('GW40_RAW_SOURCE_ROOT', ARTIFACTS/'raw_dualxf')
SOURCE_MANIFEST = RAW / 'RAW_DUALXF_RMS_EXEC_20261020/inputs/DATA_MANIFEST.json'
HARD = ['DNA_RENDERING/0097/0097_04/0000', 'DNA_RENDERING/0019/0019_06/0224']


def read_json(path):
    """读取已有收据，绝不将生成图作为源图。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """仅写本轮新输出。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def source_batch(source, fitted=None):
    """源世界坐标不缩放；仅先验拟合条件替换既有合法顶点。"""
    with np.load(source, allow_pickle=False) as z:
        keys = ['rgb','mask','K','R','t']
        if fitted is None:
            keys += ['smplx_xyz','smplx_faces','smplx_xyz_canonical','smplx_normals_canonical']
        v = {k: z[k] for k in keys}
    if fitted:
        with np.load(fitted, allow_pickle=False) as z:
            for k in ['smplx_xyz','smplx_faces','smplx_xyz_canonical','smplx_normals_canonical']:
                v[k] = z[k]
    E = np.broadcast_to(np.eye(4, dtype=np.float32), (6,4,4)).copy()
    E[:,:3,:3], E[:,:3,3] = v['R'], v['t']
    xyz = v['smplx_xyz']
    return dict(images=v['rgb'][None], masks=v['mask'][:,0][None], K=v['K'][None], E=E[None],
                smpl_vertices=xyz[None], smpl_faces=v['smplx_faces'],
                smpl_canonical=v['smplx_xyz_canonical'][None], smpl_normals=v['smplx_normals_canonical'][None],
                unit_center=((xyz.min(0)+xyz.max(0))*.5)[None],
                unit_scale=np.asarray([np.ptp(xyz, axis=0).max()*1.3], np.float32))


def source_targets(source):
    """独立标签读取端保持原传感器投影 Z-depth 和有效支持，不除 RMS。"""
    with np.load(source, allow_pickle=False) as z:
        depth, valid = z['target_depth'][:,0], z['target_depth_valid'][:,0]
    valid = valid & np.isfinite(depth) & (depth > 0)
    return dict(depth=np.where(valid, depth, 0)[None], depth_valid=valid[None])


def source_metadata(row):
    """由实际参考源的原归档恢复相机身份，隔离另一轮恢复的旧元数据。"""
    lineage=json.loads(json.dumps(row['source_lineage']))
    ids=row['source_view_ids']
    original=lineage.get('reference_copy_record',{}).get('original_source_absolute_path')
    if original:
        with np.load(original,allow_pickle=False) as archived,np.load(row['source_path'],allow_pickle=False) as actual:
            if not (np.array_equal(archived['R'],actual['R']) and np.array_equal(archived['t'],actual['t'])):
                raise ValueError('Reference source camera changed; original archive identities cannot label this input.')
            ids=[f'{int(value):02d}' for value in archived['camera_ids']]
        if ids != [f'{int(value):02d}' for value in row['source_view_ids']]:
            lineage['superseded_other_input_metadata']={key:lineage.pop(key) for key in
                ['fixed_source_restoration','restoration_reads','K_source','RGB_features','normalization'] if key in lineage}
            lineage['superseded_other_input_metadata']['source_view_ids']=row['source_view_ids']
        lineage['native_rgb_camera_ids']=ids
        lineage['source_view_ids']=[int(value) for value in ids]
        lineage['actual_identity_evidence']=dict(archive=original,R_t_exact=True,
            rule='Original archive camera_ids and unchanged reference R/t; no nearest-camera matching.')
    return ids,lineage


def materialize(row, fitted=None):
    """成套写输入和标签；Xf仅引用相同原 RGB 的冻结编码。"""
    sid = row['sample_id']
    role = 'SOURCE_FITTED' if fitted else 'GIVEN_PRIOR'
    folder = OUT / 'inputs' / sid.replace('/', '__') / role
    folder.mkdir(parents=True, exist_ok=True)
    inp, target = folder/'INPUTS.npz', folder/'TARGETS.npz'
    np.savez_compressed(inp, **source_batch(row['source_path'], fitted))
    np.savez_compressed(target, **source_targets(row['source_path']))
    actual_ids,actual_lineage=source_metadata(row)
    lineage = dict(sample_id=sid, source_path=row['source_path'], source_view_ids=actual_ids,
        source_lineage=actual_lineage, preprocess=row['preprocess'], prior=role, fitted_path=fitted,
        world='original calibrated world; identity transform; camera-Z in source metres',
        source_to_model_conversion_count=0, canonical_extra_scale_count=0,
        DPT_depth='B network multiplies dimensionless prediction by centered source-camera RMS once',
        target_depth='original sensor-projected camera-Z; no RMS division',
        camera='original cached K/R/t paired with depth; no half-pixel-only replacement',
        Xf_RGB=row['source_path']+'::rgb', snapshots=row['old_xf_path'],
        uncertainty='Kinect missing region remains unknown; given prior is multiview annotation, not six-source fitting')
    write_json(folder/'LINEAGE.json', lineage)
    return dict(id=sid, domain='DNA' if row['domain']=='DNA_RENDERING' else 'TH', input_npz=str(inp), target_npz=str(target),
                snapshots=row['old_xf_path'], lineage=str(folder/'LINEAGE.json'), prior_role=role,
                source_path=row['source_path'], source_view_ids=actual_ids)


def prepare():
    """装配真实 DNA24/12身份和原 TH4；固定评价与源拟合条件分栏。"""
    source = read_json(SOURCE_MANIFEST)
    dna = [materialize(r) for r in source['train'] if r['domain']=='DNA_RENDERING']
    th_path = ROOT/'research/mentor_human_model/results/TRAIN4_DATA/MANIFEST.json'
    th_source = read_json(th_path)
    th = []
    for row in th_source['samples']:
        r = dict(row, domain='TH', prior_role='GIVEN_PRIOR')
        for key in ['input_npz','target_npz','oracle_npz','query_cameras']:
            r[key] = str((th_path.parent/r[key]).resolve())
        th.append(r)
    fixed = [materialize(r) for r in source['fixed16']]
    hard = [next(r for r in dna if r['id']==sid) for sid in HARD]
    fitted = []
    for sid in HARD:
        p = (RAW/'RAW_MEETING_FOLLOWUP_20261021/prior_review/0019_corrected_source_fitted/SOURCE_FITTED.npz'
             if '/0019/' in sid else RAW/'RAW_DUALXF_RMS_EXEC_20261020/inputs/source_pose'/sid.replace('/','__')/'SOURCE_FITTED.npz')
        row = next(r for r in source['train'] if r['sample_id']==sid)
        fitted.append(materialize(row, str(p)))
    manifest = dict(train=dna+th, fixed16=fixed, hard=hard, source_fitted=fitted,
        th_control=[r for r in th if '/0004/' in r['id']],
        dna24_identity_count=len({r['id'].split('/')[1] for r in dna}),
        accepted_input_schemes=1, input_scheme='original matched RGB/mask/K/E/Kinect grid, B world units',
        existing_source_manifest=str(SOURCE_MANIFEST), RAW_model_consumed=False,
        new_DNA_Kinect_training_not_historical_DA3=True,
        historically_exposed_diagnostic='DNA_RENDERING/0012/0012_09/0149',
        new_teacher_forwards=0, new_source_pose_fits=0, new_RGB_encoding_required=False)
    write_json(OUT/'inputs/SAMPLE_MANIFEST.json', manifest)
    write_json(OUT/'inputs/TRAIN_MANIFEST.json', dict(samples=dna+th, source_manifest=str(OUT/'inputs/SAMPLE_MANIFEST.json')))
    write_json(OUT/'inputs/EVAL_MANIFEST.json', dict(samples=fixed+hard+fitted+manifest['th_control'], source_manifest=str(OUT/'inputs/SAMPLE_MANIFEST.json')))
    print(json.dumps(dict(dna=len(dna), th=len(th), fixed=len(fixed), fitted=len(fitted))))


if __name__ == '__main__':
    prepare()
