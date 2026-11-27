"""显式样本到已接受教师的索引，以及原上限内的两支200更新合同。"""
import json
from pathlib import Path
import torch
from research.dna_b_adapter_20261021.runtime import ROOT
from research.dna_b_adapter_20261021.paired_train import balanced_plan, load_arrays, write_json
from research.dna_b_adapter_20261021.teacher_diagnostic import teacher_target

SCOPE = 'MULTISUBJECT_DEPTH_ONLY_200'
OLD = ROOT/'artifacts/thuman20/DNA_TEACHER_COMPARE_20261023'
B_INPUTS = ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021/inputs/SAMPLE_MANIFEST.json'


def read_json(path):
    """读取实际合同或科学接受文件，不把候选状态改为接受。"""
    return json.loads(Path(path).read_text(encoding='utf8'))


def accepted_entries(registry_path):
    """只接纳逐样本明确批准的深度监督，旧oracle许可不能自动升级。"""
    registry = read_json(registry_path)
    entries = {}
    for sid, entry in registry['teachers'].items():
        if entry['status'] != 'ACCEPTED_FOR_DEPTH_SUPERVISION':
            continue
        decision = read_json(entry['acceptance'])
        if (decision['sample'] != sid or decision['scope'] != SCOPE
                or decision['depth_supervision_allowed'] is not True
                or Path(decision['oracle_archive']).resolve() != Path(entry['oracle_archive']).resolve()
                or Path(decision['approved_valid_npz']).resolve() != Path(entry['approved_valid_npz']).resolve()):
            raise ValueError('教师样本、已消费数组或批准支持不一致: '+sid)
        entries[sid] = entry
    return entries


def approved_target(entry, batch):
    """复用原B已验证的坐标转换，仅收窄到明确批准的像素支持。"""
    target = teacher_target({'teacher_oracle_archive': entry['oracle_archive'], 'teacher_field_coefficient': 0}, batch)
    approved = load_arrays(entry['approved_valid_npz'])['valid']
    if approved.dtype != torch.bool or approved.shape != target['valid'].shape:
        raise ValueError('批准支持必须是原六源布尔网格，保留batch维度')
    if (approved & ~target['valid']).any() or not approved.any():
        raise ValueError('批准支持为空或超出真实教师可见前景')
    target['valid'] = approved
    return target


def prepare_contract(registry_path, output):
    """冻结24DNA和4TH原曝光，两支共用U600权重、Adam、RNG及唯一200端点。"""
    entries = accepted_entries(registry_path)
    samples = read_json(B_INPUTS)['train']
    dna = [row for row in samples if row['domain'] == 'DNA']
    th = [row for row in samples if row['domain'] == 'TH']
    if len(dna) != 24 or len(th) != 4:
        raise ValueError('本合同必须保留原24DNA与4TH，不缩成教师样本子集')
    dna_ids = {row['id'] for row in dna}
    if not set(entries) <= dna_ids or len({sid.split('/')[1] for sid in entries}) < 2:
        raise ValueError('新教师必须属于真实训练集且至少覆盖两个身份')
    if not any(row['new_supervision_this_run'] is True for row in entries.values()):
        raise ValueError('没有新增合法监督时不能把旧48步诊断重复当新实验')
    binding = read_json(OLD/'B_STUDENT_INTERFACE_BINDING.json')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/'TRAINING_CONTRACT.json'
    if path.exists():
        raise FileExistsError('冻结合同已存在，不能隐式改曝光、接受支持或预算')
    contract = dict(scope=SCOPE, branches=['K', 'T'], updates_per_branch=200,
        total_update_budget=400, initialize_from=binding['checkpoint'],
        optimizer_options=binding['fresh_optimizer_options'],
        optimizer_initialization='LOAD_U600_SAVED_STATE',
        rng_initialization='RESTORE_U600_SAVED_PYTHON_NUMPY_TORCH_CUDA_AFTER_PREPARATION',
        samples=samples, draw_plan=balanced_plan(samples, 20261016, 200), output=str(out),
        teacher_registry=str(Path(registry_path).resolve()), teachers=entries,
        teacher_depth_coefficient=1.0, teacher_field_coefficient=0.0, teacher_normal_coefficient=0.0,
        sample_weight=.25, teacher_support='Explicit approved valid subset of first-surface visible source foreground; no Kinect-error filtering',
        no_teacher_policy='Keep original Kinect target and exposure; teacher coefficient zero',
        student_prior='Original GIVEN_PRIOR inputs; source-fitted prior only bounds teacher construction',
        frozen=['vggt.aggregator', 'appearance'], camera_head='absent; original reference K/E',
        formal_training_allowed=True, endpoint_updates=[200], checkpoint_updates=[100,200], old_diagnostic_updates_reused_as_start=False,
        restart_policy='Stop on actual failure; no automatic restart or budget extension',
        adoption='Requires fixed DNA identity evaluation, TH retention, complete reference RAW/mesh and real held-out views; no native Camera Head output; no endpoint cherry-picking')
    write_json(path, contract)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(prepare_contract(args.registry, args.output))
