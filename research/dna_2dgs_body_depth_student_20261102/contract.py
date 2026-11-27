"""单0019探索性伪Depth的已授权合同；不修改旧多教师合同。"""
import json
from pathlib import Path
import numpy as np
from research.dna_b_adapter_20261021.paired_train import balanced_plan, load_arrays, write_json
from research.dna_b_adapter_20261021.teacher_supervision import approved_target

ROOT = Path(__file__).resolve().parents[2]
from research.dna_gw40_dngw0019_replay_20261125.paths import RESULTS

OUT = RESULTS/'DNA_2DGS_BODY_DEPTH_STUDENT_20261102'
SOURCE = RESULTS/'DNA_GAUSSIAN_MESH_MENTOR_20261102/body_pseudodepth_diagnostic'
SID = 'DNA_RENDERING/0019/0019_06/0224'
OLD = ROOT/'artifacts/thuman20/DNA_MULTISUBJECT_REPAIR_20261024/student/formal_KT200'
SCOPE = 'SINGLE_0019_EXPLORATORY_2DGS_BODY_DEPTH_200'


def read(path):
    """读取既有证据原件。"""
    return json.loads(Path(path).read_text(encoding='utf8'))


def prepare():
    """一次核对实际支持及旧K共同条件，冻结唯一新增T200。"""
    old = read(OLD/'TRAINING_CONTRACT.json')
    prior = read(RESULTS/'DNA_NEUS2_CROSS_ID_AND_STUDENT_20261026/student_neus2_T200/TRAINING_CONTRACT.json')
    unchanged = ['initialize_from','optimizer_options','optimizer_initialization','rng_initialization',
                 'samples','draw_plan','sample_weight','frozen','camera_head','student_prior',
                 'updates_per_branch','checkpoint_updates']
    equal = {key: old[key] == prior[key] for key in unchanged}
    samples = read(ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021/inputs/SAMPLE_MANIFEST.json')['train']
    if samples != old['samples'] or balanced_plan(samples, 20261016, 200) != old['draw_plan']:
        raise ValueError('当前原28样本或原draw改变，不可复用K')
    run = read(OLD/'K/RUN.json')
    if not all(equal.values()) or run['actual_updates'] != 200 or run['parent'] != old['initialize_from']:
        raise ValueError('旧K实际共同条件不一致，必须报告而非静默复用')
    if read(OLD/'K/DRAW_PLAN.json') != old['draw_plan']:
        raise ValueError('旧K实际draw不匹配')
    item = next(row for row in samples if row['id'] == SID)
    entry = dict(oracle_archive=str(SOURCE/'DEPTH_ONLY_DIAGNOSTIC_INPUT.npz'),
                 approved_valid_npz=str(SOURCE/'CANDIDATE_VALID_DIAGNOSTIC.npz'),
                 use='USER_AUTHORIZED_EXPLORATORY_PSEUDODEPTH_NOT_CERTIFIED_GT')
    batch = load_arrays(item['input_npz']); target = load_arrays(item['target_npz'])
    teacher = approved_target(entry, batch)
    with np.load(entry['approved_valid_npz']) as z:
        original_m = z['original_M'] if 'original_M' in z.files else None
    valid = teacher['valid']; kval = target['depth_valid'].bool()
    if int(valid.sum()) != 67936 or int(kval.sum()) != 70738 or bool((valid & ~kval).any()):
        raise ValueError('实际固定67936/70738支持改变')
    if not bool((teacher['depth'][valid] > 0).all()) or not bool((batch['unit_scale'] > 0).all()):
        raise ValueError('实际teacher深度或原尺度不合法')
    exposures = sum(SID in draw['DNA'] for draw in old['draw_plan'])
    if exposures != 17: raise ValueError('0019原曝光不是17')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'TRAINING_CONTRACT.json'
    if path.exists(): raise FileExistsError('不覆盖已冻结合同')
    contract = {key: old[key] for key in unchanged}
    contract.update(scope=SCOPE, output=str(OUT), branches=['T'], teachers={SID:entry},
        total_update_budget=400, planned_new_updates=200, control_new_updates=0,
        control_checkpoint=str(OLD/'K/checkpoint_u0200.pt'), teacher_depth_coefficient=1.,
        teacher_normal_coefficient=0., teacher_field_coefficient=0., E1=False,
        formal_training_allowed=True, certified_GT=False, teacher_support='Original M & body_hit & finite_positive_depth; no U, error/region filtering',
        authorization='Parent relayed explicit user approval for exploratory T200 and necessary C200; max400',
        source_draft=str(SOURCE/'EXPLORATORY_PSEUDODEPTH_CONTRACT_DRAFT_CN.md'))
    write_json(path, contract)
    write_json(OUT/'INPUT_AND_CONTROL_REUSE_RECEIPT.json',dict(status='MATCHED_OLD_K200_REUSED',
        unchanged_contract_fields=equal, samples_equal_current_manifest=True, actual_old_draw_equal=True,
        old_K_RUN=str(OLD/'K/RUN.json'), old_K_checkpoint=contract['control_checkpoint'],
        prior_unchanged_science=str(RESULTS/'DNA_NEUS2_CROSS_ID_AND_STUDENT_20261026/SCIENCE_ACTUAL_STUDENT_PATH_REVIEW.md'),
        teacher_valid=int(valid.sum()), kinect_valid=int(kval.sum()), teacher_per_source=valid.sum((0,2,3)).tolist(),
        unit_scale=batch['unit_scale'].tolist(), actual_teacher_exposures=exposures,
        K_E_cube_full_foreground='Exact tensor equality by original teacher_target',
        draft_files_unchanged=True, certified_GT=False, camera17='MASK_SEMANTICS_UNRESOLVED', new_C_updates=0))
    return path
