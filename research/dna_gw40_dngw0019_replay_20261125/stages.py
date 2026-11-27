"""统一入口的逐阶段实现。"""

from pathlib import Path
import sys

import numpy as np

from replay import (BODY, NEW_FRAMES, OFFICIAL, OUT, PREVIOUS, RESULTS, ROOT, SOURCE, STUDENT,
                    decision_for, load, row_for, sample, save, student_farm)


def control():
    """直接读取原 0019 两份目标、三个相机集合和实际 draw。"""
    import matched_student as matched

    row = row_for('0019')
    frozen = load(STUDENT/'TRAINING_CONTRACT.json')
    original_input = RESULTS/'DNA_GAUSSIAN_TEACHER_EXEC_20261102/inputs/DNA0019_INPUT.json'
    cameras = load(original_input)
    body_path = OFFICIAL/'human_body_only/BODY_ONLY_RECEIPT.json'
    body = load(body_path)
    target = load(frozen['teacher_target_receipt'])
    branch = load(STUDENT/'DN_GW/RUN.json')
    plan = matched.matched_plan(row['id'])
    *_, original_measured = matched.input_support(row)
    if (cameras['sample'] != row['id'] or sorted(cameras['train_ids']+cameras['held_ids']) != list(range(48))
            or len(cameras['train_ids']) != 40 or not set(BODY) <= set(cameras['train_ids'])
            or [r['camera_id'] for r in body['selection']] != BODY
            or Path(target['mesh']) != body_path.with_name('GW_HUMAN_BODY_ONLY_DISPLAY.ply')
            or branch['actual_updates'] != 200 or plan != load(STUDENT/'DN_GW/DRAW_PLAN.json')):
        raise ValueError('0019 原相机、网格或训练实例不符')
    with (np.load(frozen['teacher_target_archive']) as t, np.load(frozen['source_V']) as v,
          np.load(row['input_npz']) as inp, np.load(target['depth_source']) as z,
          np.load(target['face_index']) as face):
        mask, normal = t['valid'], t['normal_valid']
        selected = (original_measured & np.isin(SOURCE, matched.ALLOWED)[None,:,None,None]
                    & z['valid'][None] & (z['depth'][None] > 0) & np.isfinite(z['depth'][None]))
        if (not np.array_equal(mask, selected) or not np.array_equal(mask, v['valid']) or (normal & ~mask).any()
                or not np.array_equal(t['K'], inp['K'][0]) or not np.array_equal(t['E'], inp['E'][0])
                or not np.array_equal(t['V_face_index'], face['face_id'][mask[0]])
                or not np.array_equal(t['depth'][mask], z['depth'][mask[0]])
                or (int(mask.sum()), int(normal.sum())) != (target['depth_count'], target['normal_count'])):
            raise ValueError('历史目标 K/E、深度、索引或支持不符')
    prefix = PREVIOUS/'PARITY_0019'
    parity = load(prefix/'PARITY_READBACK.json')
    inference = load(Path(parity['replay'])/'INFERENCE.json')
    if (inference['id'] != row['id'] or Path(inference['checkpoint']) != STUDENT/'DN_GW/checkpoint_u0200.pt'
            or inference['input_npz'] != row['input_npz'] or inference['frozen_Xf'] != row['snapshots']):
        raise ValueError('历史控制前向与冻结行不同')
    save(OUT/'control_0019/BOUND.json', dict(
        sample=row['id'], consumer_bind='ORIGINAL_ARCHIVED_ARRAYS_READ', original_input=str(original_input),
        source_camera_ids=SOURCE, train=cameras['train_ids'], held=cameras['held_ids'],
        body=str(body_path), target=frozen['teacher_target_archive'], original_target_source=frozen['source_V'],
        original_depth_pixels=int(mask.sum()), original_normal_pixels=int(normal.sum()), transferred_V_diff_pixels=0,
        actual_draw_plan=str(STUDENT/'DN_GW/DRAW_PLAN.json'),
        original_actual_draws=str(STUDENT/'DN_GW/updates.jsonl'), original_student_updates=branch['actual_updates'],
        original_teacher_exposures=branch['teacher_exposures'], original_u600=frozen['initialize_from'],
        original_dn_u200=inference['checkpoint'], zero_update_replay=parity['replay'],
        previous_numeric_comparison=str(prefix/'PARITY_READBACK.json'),
        display_receipt=str(STUDENT.parent/'display_dense/RAW_BODY_DENSE_RGB/RECEIPT.json'),
        new_network_forwards=0, new_optimizer_updates=0))


def evaluate(code, farm):
    """用新检查点评估原固定 18 例和本例，已有完成帧按实际绑定续接。"""
    from research.dna_b_adapter_20261021.evaluate import evaluate as original
    fixed = STUDENT.parent/'evaluation/ENDPOINT18_MANIFEST.json'
    samples = list(load(fixed)['samples'])
    row = row_for(code)
    samples = [row] + [r for r in samples if r['id'] != row['id']]
    checkpoint = farm/'DN_GW/checkpoint_u0200.pt'
    dest = farm/'evaluation/DN_GW_U200'
    pending = []
    receipts = []
    for item in samples:
        path = dest/item['id'].replace('/', '__')/item['prior_role']/'INFERENCE.json'
        receipts.append(path)
        if path.is_file():
            actual = load(path)
            if (Path(actual['checkpoint']) != checkpoint or actual['id'] != item['id']
                    or actual['input_npz'] != item['input_npz'] or actual['frozen_Xf'] != item['snapshots']):
                raise ValueError(f'{path} 与冻结输入不同')
        else:
            pending.append(item)
    if pending:
        manifest = farm/'evaluation/REMAINING_INPUTS.json'
        save(manifest, dict(source=str(fixed), samples=pending))
        original(checkpoint, manifest, dest)
    rows = [load(path) for path in receipts]
    save(dest/'EVALUATION_RECEIPT.json', rows)
    for path, item in zip(receipts, rows):
        folder = path.parent
        files = [folder/'source_RAW.npz', folder/'source_RAW_all_foreground.ply',
                 *(folder/f'source_{n:02d}/render.npz' for n in range(6))]
        if item['surface_status'] == 'EXPORTED_ALL_COMPONENTS':
            files.extend([folder/'base_mesh.ply', folder/'clothed_mesh.ply'])
        for actual in files:
            if not actual.is_file():
                raise FileNotFoundError(actual)
    from retention import compare
    compare(code)


def display(code, farm):
    """沿用原 2M 同源插值及逐点父索引，只换成该帧实际 U200。"""
    import batch_body_display as exporter

    sid = sample(code)
    case = farm/'evaluation/DN_GW_U200'/sid.replace('/', '__')
    exporter.DEST = farm/'display_2M'
    folder = exporter.DEST/case.name
    checkpoint = farm/'DN_GW/checkpoint_u0200.pt'

    def load_matched(case_path):
        source = case_path/'GIVEN_PRIOR'
        actual = load(source/'INFERENCE.json')
        row = row_for(code)
        if (actual['id'] != sid or Path(actual['checkpoint']) != checkpoint
                or Path(actual['output']) != source or actual['input_npz'] != row['input_npz']
                or actual['frozen_Xf'] != row['snapshots']):
            raise ValueError('显示源与实际匹配训练检查点不同')
        with np.load(source/'source_RAW.npz') as z, np.load(row['input_npz']) as batch:
            data = {key:z[key][0] for key in ('world_points','mask','depth','K','E')}
            rgb = np.clip(batch['images'][0].transpose(0,2,3,1)*255, 0, 255).astype('u1')
            if not all(np.array_equal(data[key], batch[name][0]) for key,name in
                       [('mask','masks'),('K','K'),('E','E')]):
                raise ValueError('显示相机或前景不同')
        fg = data['mask'].astype(bool)
        data['mask'] = fg
        raw = exporter.read_vertices(source/'source_RAW_all_foreground.ply', exporter.COLORED)
        if not np.array_equal(raw['xyz'], data['world_points'][fg]) or not np.array_equal(raw['rgb'], rgb[fg]):
            raise ValueError('RAW PLY 与原像素网格不同')
        index = np.full(fg.shape, -1, dtype='i4')
        index[fg] = np.arange(len(raw), dtype='i4')
        return source, actual, load(row['lineage']), raw, data, rgb, index

    exporter.load_source = load_matched
    if not (folder/'COMPONENT_REVIEW.json').is_file():
        exporter.audit_case(case)
    selection = folder/'DISPLAY_COMPONENT_SELECTION.json'
    # 仅有本帧、此检查点的明确地面块审阅才删除点。
    decisions = dict(components=[], review='Retain all original points; no detached floor component selected.')
    if selection.is_file():
        reviewed = load(selection)
        if reviewed['sample'] != sid or Path(reviewed['checkpoint']) != checkpoint:
            raise ValueError('地面块审阅不属于当前检查点')
        decisions = reviewed
    exporter.export_case(case, decisions)


def main(action, code, base):
    import frame_gw40 as gw
    import matched_student as matched

    run = base/f'GW40_{code}'
    farm = student_farm(code)
    matched.RESULT = base
    if action == 'control':
        return control()
    if action == 'source':
        from diagnostic import read_student_source
        return read_student_source(code)
    if action == 'teacher_endpoint':
        from read_endpoint import read_endpoint
        return read_endpoint(run)
    if action == 'prepare':
        from bindings import teacher_input
        gw.inputs = teacher_input
        return gw.prepare(code, run)
    if action in ('teacher','body'):
        return getattr(gw, action)(run)
    if action in ('qa', 'freeze_qa'):
        if code == '0102':
            import teacher_gw40_qa as qa
        else:
            from bindings import qa_input
            import teacher_replay_qa as qa
            qa.input_contract = lambda frame_code: qa_input(frame_code, qa)
        qa.OUT = base
        return qa.run(code, action == 'freeze_qa')
    # 历史教师/目标保留原绝对路径，新的学习器单独写入统一结果目录。
    row = row_for(code)
    frame = load(run/'FRAME_INPUT.json')
    qa = base/'GW40_TEACHER_QA'/code
    frozen = load(qa/'FROZEN_TEACHER_QA.json')
    old = load(STUDENT/'TRAINING_CONTRACT.json')
    if (row['id'] != frame['sample'] or row['id'] != frozen['sample']
            or frame['train_ids'] != frozen['train'] or frame['held_ids'] != frozen['held']
            or frozen['six_student_sources'] != SOURCE or row['input_npz'] != frozen['original_inputs']
            or row['target_npz'] != frozen['original_targets']):
        raise ValueError('同帧教师与学生原六源绑定不同')
    if action in ('targets','bind_targets'):
        from bindings import student_camera_errors
        teacher_cameras = sorted(frame['rows'], key=lambda r:r['camera_id'])
        student_camera_errors(row, teacher_cameras)
    farm.mkdir(parents=True, exist_ok=True)
    matched.refs = lambda unused: (run, farm, row['id'], old, row)

    def admission(case, contract):
        path = decision_for(case, base)
        verdict, summary = load(path), load(qa/'TEACHER_QA_SUMMARY.json')
        proof = load(qa/'ORIGINAL_SUBSET_READBACK.json')
        target = load(contract['teacher_target_receipt'])
        mesh = run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
        allowed = (verdict['authorized_for_isolated_diagnostic'] if case in NEW_FRAMES
                   else verdict['admitted_for_matched_reproduction'])
        if (allowed is not True or verdict['sample'] != row['id']
                or summary['sample'] != row['id'] or any(Path(p) != mesh for p in
                [verdict['mesh'], summary['meshes']['BODY'], target['mesh']])
                or not proof['triangle_original_mapping_equal'] or not proof['point_world_xyz_rgb_equal']):
            raise ValueError('目标与已准入 BODY 网格不一致')
        return path

    matched.check_admission = admission
    if action == 'bind_targets':
        original_contract = run/'matched_student/TRAINING_CONTRACT.json'
        contract = load(original_contract)
        plan = matched.matched_plan(row['id'])
        if contract['student_input'] != row or contract['draw_plan'] != plan:
            raise ValueError('已冻结目标的训练行或 draw 不同')
        admission_path = admission(code, contract)
        contract.update(output=str(farm), original_matched_contract=str(original_contract),
                        admission=str(admission_path))
        save(farm/'TRAINING_CONTRACT.json', contract)
        save(farm/'DRAW_PLAN.json', plan)
        return matched.verify_targets(code, contract)
    if action == 'targets':
        import shutil
        for name in ('FOUR_FRAME_FROZEN_INPUTS.json','GW40_TARGET_SELECTION_RULE.json'):
            if not (base/name).exists():
                shutil.copyfile(PREVIOUS/name, base/name)
            if load(base/name) != load(PREVIOUS/name):
                raise ValueError(f'冻结合同不同 {base/name}')
        contract = matched.configure(code)
        matched.raster_targets(code)
        return matched.verify_targets(code, contract)
    if action == 'check':
        return matched.learn(code, 'check')
    if action == 'student':
        from research.dna_gw_geometry_student_20261103 import train as history
        from first_slot_train import train
        contract = load(farm/'TRAINING_CONTRACT.json')
        matched.check_admission(code, contract)
        history.OUT, history.SID = farm, sample(code)
        ledger = farm/'DN_GW/RUN.json'
        if ledger.exists():
            state = load(ledger)['status']
            if state == 'STOPPED_REAL_ERROR' and not (farm/'DN_GW/RECOVERY.json').exists():
                train(contract, 'DN_GW', farm, recover=True)
            elif state != 'COMPLETE_EXPLORATORY_GW200':
                raise RuntimeError(f'请检查当前学生状态 {ledger}: {state}')
        else:
            train(contract, 'DN_GW', farm)
        return matched.verify_run(farm, contract)
    if action == 'evaluate':
        return evaluate(code, farm)
    if action == 'display':
        return display(code, farm)
    raise ValueError(action)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
