"""从逐帧回执续接 0019 GW40 → DN_GW。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from research.dna_gw40_dngw0019_replay_20261125.paths import (ARTIFACTS, OFFICIAL, OUTPUT, QA_PYTHON, RASTER, RASTER_PYTHON, REPO,
                                                          RESULTS, RUNTIME, TEACHER_PYTHON, VIT_PYTHON)

ROOT = REPO
OUT = OUTPUT
PREVIOUS = RESULTS / 'DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125'
STUDENT = RESULTS / 'DNA_GW_GEOMETRY_STUDENT_20261103/student'
RENDER = STUDENT.parent / 'teacher_neighbor_fix'
FRAMES = {
    '0019': ('0019_06', '0224'), '0102': ('0102_02', '0000'),
    '0008': ('0008_01', '0000'), '0047': ('0047_01', '0000'),
    '0097': ('0097_04', '0000'), '0121': ('0121_02', '0135'),
    '0034': ('0034_04', '0000'),
}
SOURCE = [2, 11, 17, 26, 35, 41]
BODY = [2, 26, 11, 35]
NEW_FRAMES = ('0097', '0121', '0034')
sys.path.insert(0, str(RUNTIME))


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def load(path):
    return json.loads(read(path))


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sample(code):
    seq, frame = FRAMES[code]
    return f'DNA_RENDERING/{code}/{seq}/{frame}'


def row_for(code):
    sid = sample(code)
    manifest = (STUDENT.parent / 'evaluation/ENDPOINT18_MANIFEST.json' if code == '0034'
                else STUDENT / 'TRAINING_CONTRACT.json')
    row = next(row for row in load(manifest)['samples'] if row['id'] == sid)
    if list(map(int, row['source_view_ids'])) != SOURCE or row['prior_role'] != 'GIVEN_PRIOR':
        raise ValueError(f'{sid} 原始六源或 prior 不一致')
    for key in ('input_npz', 'target_npz', 'snapshots', 'lineage'):
        if not Path(row[key]).is_file():
            raise FileNotFoundError(row[key])
    return row


def read48(code):
    base = ARTIFACTS / 'DNA_MULTISUBJECT_REPAIR_20261024/inputs' / sample(code).replace('/', '__')
    split_path = base / 'SPLIT40_8.json'
    receipts = list(base.glob('*READ48_RECEIPT.json'))
    if code in NEW_FRAMES and (not split_path.is_file() or len(receipts) != 1):
        base = OUT/f'GW40_{code}'/'role_inputs'
        split_path = base/'SPLIT40_8.json'
        receipts = list(base.glob('*READ48_RECEIPT.json'))
    if not split_path.is_file() or len(receipts) != 1:
        return None
    roles, source = load(split_path), load(receipts[0])
    train, held = roles['train'], roles['heldout_sorted']
    records = sorted(source['records'], key=lambda r: r['camera_id'])
    if (roles['sample_id'] != sample(code) or source['sample_id'] != sample(code)
            or len(train) != 40 or len(held) != 8 or sorted(train + held) != list(range(48))
            or not set(SOURCE + BODY) <= set(train) or list(map(int, roles['source_view_ids'])) != SOURCE
            or [r['camera_id'] for r in records] != list(range(48))
            or any(r['frame'] != int(FRAMES[code][1]) for r in records)):
        raise ValueError(f'{sample(code)} 相机角色或原始物理帧不同')
    return split_path, receipts[0], train, held, records


def bound_frame(code, run, source):
    actual = load(run / 'FRAME_INPUT.json')
    split, receipt, train, held, rows = source
    if (actual['sample'] != sample(code) or actual['train_ids'] != train or actual['held_ids'] != held
            or Path(actual['role_receipt']) != split or Path(actual['source_receipt']) != receipt
            or len(actual['rows']) != len(rows)):
        raise ValueError(f'{run} 实际输入与逐帧原始回执不同')
    for original, used in zip(rows, actual['rows']):
        if any(used[k] != original[k] for k in ('camera_id', 'rgb', 'mask', 'K', 'E', 'frame')):
            raise ValueError(f'{run} 相机 {original["camera_id"]} 绑定不同')


def student_farm(code):
    return OUT/f'GW40_{code}'/'matched_student'


def decision_for(code, base):
    if code in NEW_FRAMES:
        return base/'GW40_TEACHER_QA'/code/'DIAGNOSTIC_AUTHORIZATION.json'
    historical = base/'GW40_TEACHER_QA'/code/'TEACHER_ADMISSION.json'
    current = OUT/'GW40_TEACHER_QA'/code/'MANUAL_ADMISSION.json'
    if not current.is_file():
        return historical
    manual, prior = load(current), load(historical)
    mesh = base/f'GW40_{code}'/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    if (manual['sample'] != sample(code) or manual['admitted_for_matched_reproduction'] is not True
            or Path(manual['mesh']) != mesh or Path(manual['original_QA_decision']) != historical
            or prior['sample'] != manual['sample'] or Path(prior['mesh']) != mesh
            or manual['original_QA_status'] != prior['status']):
        raise ValueError('人工裁定与独立 QA 对应的帧或网格不同')
    return current


def stage(name, code, base):
    """按已验证的 Python/CUDA 环境执行该阶段。"""
    qa_stage = name in ('source', 'teacher_endpoint', 'qa', 'freeze_qa', 'display')
    python = QA_PYTHON if qa_stage else VIT_PYTHON
    env = os.environ.copy()
    env.update(KMP_DUPLICATE_LIB_OK='TRUE', PYTHONUTF8='1')
    env['PATH'] = os.pathsep.join(map(str, [python.parent/'Library/bin', python.parent/'Scripts', env['PATH']]))
    if name in ('teacher', 'targets'):
        python = TEACHER_PYTHON if name == 'teacher' else RASTER_PYTHON
        native_root = OFFICIAL if name == 'teacher' else RASTER
        env.update(GW40_NATIVE_PYTHON=str(python), GW40_NATIVE_JIT=str(native_root/'native_jit'))
        subprocess.run([str(HERE/'native_stage.cmd'), name, code, str(base)], env=env, cwd=ROOT, check=True)
        return
    subprocess.run([str(python), '-X', 'utf8', '-u', str(HERE/'stages.py'),
                    name, code, str(base)], env=env, cwd=ROOT, check=True)


def main(code, until):
    row = row_for(code)
    output = OUT/'continuations'/f'{code}.json'
    report = dict(sample=sample(code), original_input=row['input_npz'], frozen_Xf=row['snapshots'])

    def finish(status, **fields):
        report.update(status=status, **fields)
        save(output, report)
        print(f'{sample(code)}  {status}  {output}', flush=True)

    if code == '0019':
        stage('control', code, OUT)
        return finish('BOUND_ORIGINAL_0019', evidence=str(OUT/'control_0019/BOUND.json'))
    source = read48(code)
    if source is None:
        if code == '0034':
            return finish('FIXED_DEV_RAW_INPUT_UNAVAILABLE',
                          role_evidence=str(STUDENT.parent/'evaluation/ENDPOINT18_MANIFEST.json'))
        other = (ARTIFACTS/'DNA_TEACHER_COMPARE_20261023/teacher/0097_04_0000/SPLIT.json'
                 if code == '0097' else PREVIOUS/'CANDIDATE_ROLE_AUDIT.md')
        return finish('NO_BOUND_SAME_FRAME_GW40_SPLIT', available_evidence=str(other))
    # 同一帧的已完成教师继续使用原目录，避免重训或改写历史材料。
    prior = PREVIOUS/f'GW40_{code}'
    base = PREVIOUS if code not in NEW_FRAMES and (prior/'GW40_ACTUAL_ENDPOINT.json').is_file() else OUT
    run = base/f'GW40_{code}'
    qa = base/'GW40_TEACHER_QA'/code
    farm = student_farm(code)
    report.update(teacher=str(run), split=str(source[0]), rgb_and_masks=str(source[1]))
    if code in NEW_FRAMES:
        source_readback = run/'STUDENT_SOURCE_READBACK.json'
        if not source_readback.is_file():
            stage('source', code, base)
        report.update(scope='ISOLATED_DIAGNOSTIC', same_frame_evaluation='IN_SAMPLE',
                      split_role=load(source[0]).get('role', 'DIRECT_SAME_FRAME_RECEIPT'),
                      student_source_readback=str(source_readback))
    if until == 'bind':
        if (run/'FRAME_INPUT.json').is_file():
            bound_frame(code, run, source)
        return finish('BOUND_FRAME')
    run.mkdir(parents=True, exist_ok=True)
    if not (run/'CENTERED_INPUT_PROJECTION_RECEIPT.json').is_file():
        stage('prepare', code, base)
    bound_frame(code, run, source)
    if not (run/'GW40_RUN.json').is_file():
        stage('teacher', code, base)
    teacher = load(run/'GW40_RUN.json')
    if teacher['status'] != 'COMPLETED':
        return finish('TEACHER_' + teacher['status'], teacher_receipt=str(run/'GW40_RUN.json'))
    if code in NEW_FRAMES and not (run/'GW40_ACTUAL_ENDPOINT.json').is_file():
        stage('teacher_endpoint', code, base)
    if not (run/'human_body_only/BODY_ONLY_RECEIPT.json').is_file():
        stage('body', code, base)
    targets = farm/'SERIALIZED_TARGET_READBACK.json'
    if code in NEW_FRAMES and not targets.is_file():
        # 同帧目标先冻结，学生启动仍以独立验收回执为准。
        stage('freeze_qa', code, base)
        stage('targets', code, base)
    if not (qa/'TEACHER_QA_SUMMARY.json').is_file():
        stage('qa', code, base)
    if code in NEW_FRAMES:
        from diagnostic import authorize_run
        authorize_run(code, base)
    decision = decision_for(code, base)
    if not decision.is_file():
        return finish('AWAITING_TEACHER_VISUAL_REVIEW', qa=str(qa/'TEACHER_QA_SUMMARY.json'))
    verdict = load(decision)
    mesh = run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    if verdict['sample'] != row['id'] or Path(verdict['mesh']) != mesh or not mesh.is_file():
        raise ValueError('准入回执与实际教师不同')
    report.update(teacher_decision=str(decision), teacher_verdict=verdict['status'])
    admitted = (verdict['authorized_for_isolated_diagnostic'] if code in NEW_FRAMES
                else verdict['admitted_for_matched_reproduction'])
    if admitted is False:
        prior_learner = run/'matched_student/LEARNER_STAGE.json'
        if prior_learner.exists():
            old = load(prior_learner)
            if old['sample'] != row['id'] or old['optimizer_updates'] or Path(old['teacher_decision']) != decision:
                raise ValueError('原拒绝分支和学生状态不一致')
            report['previous_student_receipt'] = str(prior_learner)
        return finish('NOT_RUN_TEACHER_REJECTED', new_student_optimizer_updates=0)
    if admitted is not True:
        raise ValueError('缺正式准入')
    if until == 'teacher':
        return finish('TEACHER_QA_COMPLETE_DIAGNOSTIC' if code in NEW_FRAMES else 'TEACHER_ADMITTED',
                      qa=str(qa/'TEACHER_QA_SUMMARY.json'))
    if not targets.is_file():
        archived_targets = run/'matched_student/SERIALIZED_TARGET_READBACK.json'
        stage('bind_targets' if base == PREVIOUS and archived_targets.is_file() else 'targets', code, base)
    if until == 'targets':
        return finish('TARGETS_FROZEN', targets=str(farm/'SERIALIZED_TARGET_READBACK.json'))
    if not (farm/'FIRST_BATCH_NORMAL_GRADIENT.json').is_file():
        stage('check', code, base)
    if not (farm/'ACTUAL_DNGW_U200_READBACK.json').is_file():
        stage('student', code, base)
    checkpoint = farm/'DN_GW/checkpoint_u0200.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if until == 'student':
        return finish('MATCHED_STUDENT_U200', checkpoint=str(checkpoint))
    endpoint = farm/'evaluation/DN_GW_U200'
    published = endpoint/'EVALUATION_RECEIPT.json'
    fixed = load(STUDENT.parent/'evaluation/ENDPOINT18_MANIFEST.json')['samples']
    expected = {r['id'] for r in fixed} | {row['id']}
    existing = load(published) if published.is_file() else []
    bound = {item['id']:item for item in [*fixed, row]}
    if {r['id'] for r in existing} != expected or any(Path(r['checkpoint']) != checkpoint or
            r['input_npz'] != bound[r['id']]['input_npz'] or r['frozen_Xf'] != bound[r['id']]['snapshots']
            or not (Path(r['output'])/'source_RAW_all_foreground.ply').is_file() for r in existing):
        stage('evaluate', code, base)
    if until == 'evaluate':
        return finish('MATCHED_ENDPOINT_EVALUATED', checkpoint=str(checkpoint))
    display_receipt = farm/'display_2M'/row['id'].replace('/', '__')/'RECEIPT.json'
    existing = load(display_receipt) if display_receipt.is_file() else {}
    if existing and (existing.get('sample_id') != row['id'] or Path(existing['checkpoint']) != checkpoint
            or existing['points'] != 2000000 or not Path(existing['provenance']).is_file()
            or not Path(existing['output']).is_file() or Path(existing['output']).stat().st_size != existing['output_bytes']):
        raise ValueError(f'2M 回执与当前学生或产物不同 {display_receipt}')
    if not existing:
        stage('display', code, base)
    return finish('MATCHED_ENDPOINT_DISPLAYED', checkpoint=str(checkpoint),
                  display=str(farm/'display_2M'/row['id'].replace('/', '__')/'RECEIPT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('code', choices=FRAMES)
    parser.add_argument('--until', choices=('bind', 'teacher', 'targets', 'student', 'evaluate', 'display'),
                        default='display')
    args = parser.parse_args()
    main(args.code, args.until)
