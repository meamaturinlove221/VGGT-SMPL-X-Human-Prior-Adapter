"""以原 0019 实际抽样和原栅格投影，构造指定候选帧的 DN_GW 复现。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from research.dna_gw40_dngw0019_replay_20261125.paths import REPO as ROOT, RESULTS

RESULT = RESULTS/'DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125'
HIST = RESULTS/'DNA_GW_GEOMETRY_STUDENT_20261103/student'
OLD_SAMPLE = 'DNA_RENDERING/0019/0019_06/0224'
ALLOWED = [2, 11, 26, 35]
sys.path.insert(0, str(ROOT))


def read(path):
    """读取实际合同或观察回执。"""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    """写入本轮候选目录。"""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def refs(code):
    """将既有训练行与同帧直接输入、QA 相机和实际教师绑定。"""
    run = RESULT / f'GW40_{code}'
    farm = run / 'matched_student'
    farm.mkdir(exist_ok=True)
    frame_input = read(run / 'FRAME_INPUT.json')
    frame = frame_input['sample']
    qa = read(RESULT / 'GW40_TEACHER_QA' / code / 'FROZEN_TEACHER_QA.json')
    old = read(HIST / 'TRAINING_CONTRACT.json')
    row = next(r for r in old['samples'] if r['id'] == frame)
    frozen = next(r for r in read(RESULT / 'FOUR_FRAME_FROZEN_INPUTS.json')['samples'] if r['id'] == frame)
    if row != frozen or frame != qa['sample'] or list(map(int, row['source_view_ids'])) != qa['six_student_sources']:
        raise ValueError('原训练行、冻结输入与 QA 的样本或源相机不一致')
    if (row['input_npz'], row['target_npz']) != (qa['original_inputs'], qa['original_targets']):
        raise ValueError('学生与实际 QA 的原输入不同')
    if frame_input['train_ids'] != qa['train'] or frame_input['held_ids'] != qa['held']:
        raise ValueError('教师实际与独立 QA 的相机组不同')
    return run, farm, frame, old, row


def matched_plan(frame):
    """从 200 个实际更新恢复抽样，只替换首 DNA 槽。"""
    log = HIST / 'DN_GW/updates.jsonl'
    logged = [json.loads(line) for line in log.read_text(encoding='utf8').splitlines()]
    original = read(HIST / 'DN_GW/DRAW_PLAN.json')
    if len(logged) != 200 or len(original) != 200:
        raise ValueError('原 DN_GW 实际更新不足 200')
    plan = []
    for expected, actual in zip(original, logged):
        rows = actual['draws']
        ids = [r['id'] for r in rows]
        if (actual['update'] != expected['update'] or ids != expected['DNA'] + expected['TH']
                or ids[0] != OLD_SAMPLE or [r['domain'] for r in rows] != ['DNA', 'DNA', 'TH', 'TH']
                or [r['weight'] for r in rows] != [.25] * 4
                or any((r['teacher_depth_valid'] > 0, r['teacher_normal_valid'] > 0) != (i == 0, i == 0)
                       for i, r in enumerate(rows))):
            raise ValueError(f"原第 {actual['update']} 步实际样本或监督槽与收据不一致")
        plan.append(dict(update=actual['update'], DNA=[frame, ids[1]], TH=ids[2:]))
    return plan


def input_support(row):
    """从原 F 和 Kinect 正深度确定 M，原 loss 仍消费整份 depth_valid。"""
    with np.load(row['input_npz']) as inp, np.load(row['target_npz']) as native:
        valid = native['depth_valid'].astype(bool)
        depth = native['depth']
        measured = inp['masks'].astype(bool) & valid & np.isfinite(depth) & (depth > 0)
        return inp['K'], inp['E'], inp['unit_center'], inp['unit_scale'], valid, measured


def configure(code):
    """固定抽样、U600、优化器和原预测头，登记每帧可核验的输入。"""
    run, farm, frame, old, row = refs(code)
    k, e, center, scale, valid, measured = input_support(row)
    plan = matched_plan(frame)
    counts = Counter(sid for draw in plan for sid in draw['DNA'] + draw['TH'])
    contract = {key:old[key] for key in (
        'samples', 'initialize_from', 'optimizer_options', 'optimizer_initialization', 'rng_initialization',
        'frozen', 'camera_head', 'student_prior', 'sample_weight', 'updates_per_branch', 'checkpoints',
    )}
    if code == '0034':
        if any(item['id'] == frame for item in old['samples']):
            raise ValueError('0034/0000 出现在历史训练集，和独立诊断的原角色不符')
        contract['samples'] = [*old['samples'], row]
        contract['additional_input_role'] = 'FIXED_DEV_AS_DIAGNOSTIC_TRAINING_TARGET'
    contract.update(
        scope='MATCHED_FRAME_DN_GW_REPRODUCTION', output=str(farm), sample=frame, draw_plan=plan,
        original_actual_draws=str(HIST / 'DN_GW/updates.jsonl'), historical_source=str(HIST/'TRAINING_CONTRACT.json'),
        exposures_per_branch=dict(counts), teacher_exposures={'DN_GW':200}, total_update_budget=200,
        branches={'DN_GW':old['branches']['DN_GW']}, new_DNA_field=False, certified_GT=False,
        source_V=str(farm / 'V_FROZEN_PRE_QA_SELECTION.npz'),
        source_V_receipt=str(farm / 'V_FROZEN_PRE_QA_SELECTION.json'),
        teacher_target_receipt=str(farm / 'teacher_target/RECEIPT.json'),
        teacher_target_archive=str(farm / 'teacher_target/TARGETS.npz'),
        mask_rule_frozen=str(RESULT/'GW40_TARGET_SELECTION_RULE.json'), unit_scale=scale.tolist(),
        draw_policy='Replace the first DNA in each original logged update. The other 600 draws are unchanged.',
        original_Kinect_per_source=valid.sum((0,2,3)).tolist(), original_M_per_source=measured.sum((0,2,3)).tolist(),
        source_view_ids=row['source_view_ids'], student_input=row,
        actual_teacher_frame_input=str(run/'FRAME_INPUT.json'),
        frozen_teacher_qa=str(RESULT/'GW40_TEACHER_QA'/code/'FROZEN_TEACHER_QA.json'))
    save(farm/'TRAINING_CONTRACT.json', contract)
    save(farm/'DRAW_PLAN.json', plan)
    print(frame, '800 draws, 600 original companion slots, 200 supervised first slots;',
          counts[frame]-200, 'other original Kinect-only draws', flush=True)
    return contract


def raster_targets(code):
    """在原六视角分别调用旧深度与原三角面索引栅格，将 V 和 VN 冻结。"""
    import torch
    import nvdiffrast.torch as dr
    from research.dna_gw_geometry_student_20261103 import first_surface, targets
    sys.path.insert(0, str(ROOT / 'research/dna_b_adapter_20261021'))
    from teacher_quality import render, vertex_normals

    run, farm, frame, _, row = refs(code)
    mesh = run / 'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    body = read(mesh.parent/'BODY_ONLY_RECEIPT.json')
    if Path(body['mesh']) != mesh or body['faces'] <= 0 or body['body_mask_ids'] != [2,26,11,35]:
        raise ValueError('实际 BODY 网格与该帧构建回执不同')
    contract = read(farm / 'TRAINING_CONTRACT.json')
    targets.OUT = farm
    targets.V_PATH = Path(contract['source_V'])
    kb, eb, center, scale, _, measured = input_support(row)
    k, e = kb[0], eb[0]
    first_file = farm / 'BODY_SIX_FIRST_FACE.npz'
    old_z = farm / 'BODY_SIX_ORIGINAL_DEPTH.npz'
    if not old_z.exists():
        vertices, faces = first_surface.mesh_arrays(mesh)
        normals = vertex_normals(vertices, faces)
        v, f, n = [torch.as_tensor(arr, device='cuda') for arr in (vertices, faces, normals)]
        ctx = dr.RasterizeCudaContext()
        views = [render(v, f, n, kv, ev, *measured.shape[-2:], ctx) for kv, ev in zip(k, e)]
        np.savez_compressed(old_z, K=k, E=e, depth=np.stack([r['depth'] for r in views]),
                            valid=np.stack([r['valid'] for r in views]), mesh=str(mesh), inputs=row['input_npz'])
        del ctx, v, f, n
    if not first_file.exists():
        first_surface.render(mesh, Path(row['input_npz']), first_file)
    with np.load(old_z) as rendered:
        if (str(rendered['mesh']) != str(mesh) or str(rendered['inputs']) != row['input_npz']
                or not np.array_equal(k, rendered['K']) or not np.array_equal(e, rendered['E'])):
            raise ValueError('原深度缓存属于不同网格或相机')
        cams = list(map(int, row['source_view_ids']))
        allowed = np.isin(cams, ALLOWED)[None,:,None,None]
        v = measured & allowed & rendered['valid'][None] & (rendered['depth'][None] > 0) & np.isfinite(rendered['depth'][None])
    np.savez_compressed(targets.V_PATH, valid=v, bvyx=np.argwhere(v), source_camera_ids=cams,
                        K=kb, E=eb, unit_center=center, unit_scale=scale)
    save(Path(contract['source_V_receipt']), dict(
        sample=frame, status='FROZEN', mesh=str(mesh), camera_ids=cams, allowed=ALLOWED,
        M=measured.sum((0,2,3)).tolist(), V=v.sum((0,2,3)).tolist(), original_Kinect=row['target_npz'],
        original_input=row['input_npz'], original_render=str(old_z), rule=contract['mask_rule_frozen']))
    targets.freeze(mesh, first_file, historical_depth=old_z)


def check_admission(code, contract):
    """核对独立 QA 放行的是相同帧、相同 BODY 网格。"""
    run, farm, frame, _, row = refs(code)
    qa = RESULT / 'GW40_TEACHER_QA' / code
    decision = read(qa/'TEACHER_ADMISSION.json')
    summary = read(qa/'TEACHER_QA_SUMMARY.json')
    proof = read(qa/'ORIGINAL_SUBSET_READBACK.json')
    receipt = read(contract['teacher_target_receipt'])
    mesh = run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    if (decision['admitted_for_matched_reproduction'] is not True or frame != decision['sample']
            or frame != summary['sample'] or Path(decision['mesh']) != mesh or Path(summary['meshes']['BODY']) != mesh
            or Path(receipt['mesh']) != mesh or not proof['triangle_original_mapping_equal']
            or not proof['point_world_xyz_rgb_equal']):
        raise ValueError('本帧首面教师未取得独立 QA 的同一网格准入')
    return qa/'TEACHER_ADMISSION.json'


def verify_targets(code, contract):
    """回读冻结文件，核对原像素索引、教师深度、实际 K/E 与几何法线。"""
    run, farm, frame, _, row = refs(code)
    receipt = read(contract['teacher_target_receipt'])
    mesh = run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    k, e, _, _, original, measured = input_support(row)
    with (np.load(contract['teacher_target_archive']) as tgt, np.load(contract['source_V']) as support,
          np.load(receipt['depth_source']) as z, np.load(receipt['face_index']) as first):
        v, vn = tgt['valid'], tgt['normal_valid']
        allowed = np.isin(support['source_camera_ids'], ALLOWED)[None,:,None,None]
        expected = measured & allowed & z['valid'][None] & np.isfinite(z['depth'][None]) & (z['depth'][None] > 0)
        if (Path(receipt['mesh']) != mesh or Path(first['mesh'].item()) != mesh
                or Path(first['inputs'].item()) != Path(row['input_npz'])
                or not np.array_equal(v, expected) or not np.array_equal(v, support['valid']) or (vn & ~v).any()
                or [int(i) for i in support['source_camera_ids']] != list(map(int, row['source_view_ids']))
                or v[:,[2,5]].any() or int(v.sum()) != receipt['depth_count'] or int(vn.sum()) != receipt['normal_count']
                or not np.array_equal(tgt['v_bvyx'], np.argwhere(v)) or not np.array_equal(tgt['vn_bvyx'], np.argwhere(vn))
                or not np.array_equal(tgt['depth'][v], z['depth'][v[0]])
                or not np.array_equal(tgt['V_face_index'], first['face_id'][v[0]]) or not np.isfinite(tgt['world_normal'][vn]).all()
                or not np.array_equal(k[0], tgt['K']) or not np.array_equal(e[0], tgt['E'])):
            raise ValueError('该帧训练实际 V / VN / K / E 与原输入或冻结支持不符')
    save(farm/'SERIALIZED_TARGET_READBACK.json', dict(
        sample=frame, mesh=str(mesh), original_input=row['input_npz'],
        original_target=row['target_npz'], signed_pseudo_normal='GW original winding; no independent GT',
        K=original.sum((0,2,3)).tolist(), V=v.sum((0,2,3)).tolist(), VN=vn.sum((0,2,3)).tolist(),
        fixed_depth_targets=receipt['depth_source'], fixed_normal_triangles=receipt['face_index'],
        frozen_target=contract['teacher_target_archive'], missing_fixed_V=receipt['missing_V'],
        V_face_depth_match=receipt['V_face_match'], student_forward_count=0, optimizer_updates=0))


def learn(code, action):
    """使用原梯度检查和首槽训练循环，保留每步四个实际抽样。"""
    from research.dna_gw_geometry_student_20261103 import train as history
    from first_slot_train import train
    _, farm, frame, _, _ = refs(code)
    history.OUT, history.SID = farm, frame
    contract = read(farm/'TRAINING_CONTRACT.json')
    admission_path = check_admission(code, contract)
    if action == 'check':
        verify_targets(code, contract)
        save(farm/'MATCHED_TARGET_CONSUMPTION.json', dict(sample=frame, actual_qa=str(admission_path),
             frozen_target_readback=str(farm/'SERIALIZED_TARGET_READBACK.json'), optimizer_updates_at_binding=0))
        history.check_first_batch(contract)
    else:
        train(contract, 'DN_GW', farm)
        verify_run(farm, contract)


def stop_rejected(code):
    """将教师真实拒绝和学生未运行分别记账。"""
    run, farm, frame, _, _ = refs(code)
    path = RESULT/'GW40_TEACHER_QA'/code/'TEACHER_ADMISSION.json'
    decision = read(path)
    if (decision['sample'] != frame or decision['admitted_for_matched_reproduction'] is not False
            or Path(decision['mesh']) != run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
            or (farm/'FIRST_BATCH_NORMAL_GRADIENT.json').exists() or (farm/'DN_GW').exists()):
        raise ValueError('拒绝回执或本帧学生执行状态不同')
    save(farm/'LEARNER_STAGE.json', dict(
        sample=frame, status='NOT_RUN_TEACHER_REJECTED', teacher_decision=str(path), teacher_state=decision['status'],
        serialized_targets=str(farm/'SERIALIZED_TARGET_READBACK.json'), training_contract=str(farm/'TRAINING_CONTRACT.json'),
        student_check_forwards=0, student_training_forwards=0, optimizer_updates=0, actual_teacher_exposures=0,
        checkpoint_u200=None))


def verify_run(farm, contract):
    """逐步对照结束日志，核对 800 抽、教师作用槽和 U200。"""
    out = farm/'DN_GW'
    plan = matched_plan(contract['sample'])
    run = read(out/'RUN.json')
    target = read(contract['teacher_target_receipt'])
    actual = [json.loads(line) for line in (out/'updates.jsonl').read_text(encoding='utf8').splitlines()]
    checkpoint = out/'checkpoint_u0200.pt'
    if (not checkpoint.is_file() or run['status'] != 'COMPLETE_EXPLORATORY_GW200'
            or run['actual_updates'] != 200 or run['geometry_updates'] != 200
            or run['network_forwards'] != 800 or run['exposures'] != contract['exposures_per_branch']
            or run['teacher_exposures'] != 200 or run['teacher_normal_exposures'] != 200 or len(actual) != 200):
        raise ValueError('结束回执与固定预算不一致')
    for expected, line in zip(plan, actual):
        rows = line['draws']
        if line['update'] != expected['update'] or [r['id'] for r in rows] != expected['DNA'] + expected['TH']:
            raise ValueError('实际抽样与首槽替换合同不一致')
        for i, row in enumerate(rows):
            count = (target['depth_count'], target['normal_count']) if i == 0 else (0, 0)
            if (row['teacher_depth_valid'], row['teacher_normal_valid']) != count or row['weight'] != .25:
                raise ValueError('实际曝光或样本权重与合同不一致')
    save(farm/'ACTUAL_DNGW_U200_READBACK.json', dict(
        sample=contract['sample'], updates=200, actual_draws=800, original_companion_draws=600,
        supervised_first_draws=200, other_teacher_exposures=0, historical_actual_draws=contract['original_actual_draws'],
        actual_run=str(out/'RUN.json'), actual_draw_log=str(out/'updates.jsonl'), checkpoint=str(checkpoint),
        U0=read(farm/'ACTUAL_U0_COMPARISON_DN_GW.json')['status'], teacher_mesh=target['mesh'],
        teacher_depth_pixels=target['depth_count'], teacher_pseudo_normal_pixels=target['normal_count']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'raster', 'readback', 'stop-rejected', 'check', 'DN_GW'])
    parser.add_argument('code', choices=['0008','0047'])
    args = parser.parse_args()
    if args.action == 'prepare':
        configure(args.code)
    elif args.action == 'raster':
        raster_targets(args.code)
    elif args.action == 'readback':
        _, farm, _, _, _ = refs(args.code)
        verify_targets(args.code, read(farm/'TRAINING_CONTRACT.json'))
    elif args.action == 'stop-rejected':
        stop_rejected(args.code)
    else:
        learn(args.code, args.action)
