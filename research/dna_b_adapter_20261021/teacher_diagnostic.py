"""0097 部分教师覆盖的固定 48 更新诊断合同与原定义目标。"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from research.dna_b_adapter_20261021.runtime import ROOT
from research.dna_b_adapter_20261021.paired_train import load_arrays, write_json

OUT = ROOT/'artifacts/thuman20/DNA_TEACHER_COMPARE_20261023'
DIAG = OUT/'student_diagnostic_16x3'
TEACHER_ID = 'DNA_RENDERING/0097/0097_04/0000'


def prepare_contract():
    """冻结三组同起点同曝光合同，教师缺测保持无监督而非补零。"""
    binding = json.loads((OUT/'B_STUDENT_INTERFACE_BINDING.json').read_text(encoding='utf8'))
    oracle = OUT/'teacher/0097_04_0000/B_DPSR128_ORACLE'
    review = json.loads((oracle/'SCIENTIFIC_REVIEW.json').read_text(encoding='utf8'))
    if 'COARSE_BODY_CONSUMPTION_PASS_NO_MISSING_ARM' not in json.dumps(review):
        raise ValueError('必须先有实际完整 oracle 的粗形接受决定')
    samples = [{key: item[key] for key in ('id', 'domain', 'input_npz', 'target_npz', 'snapshots')}
               for item in binding['samples']]
    dna = [TEACHER_ID, 'DNA_RENDERING/0019/0019_06/0224']
    th = [f'THUMAN/{sid}/static/{sid}' for sid in ('0000', '0003', '0004', '0007')]
    if {s['id'] for s in samples} != set(dna+th):
        raise ValueError('诊断只允许已绑定的两 DNA 和四 TH')
    plan = [dict(update=i+1, DNA=dna, TH=th[(i % 2)*2:(i % 2)*2+2]) for i in range(16)]
    DIAG.mkdir(parents=True, exist_ok=True)
    path = DIAG/'TRAINING_CONTRACT.json'
    if path.exists():
        raise FileExistsError('合同已存在；不能隐式重写起点或预算')
    contract = dict(scope='PARTIAL_0097_TEACHER_SHORT_DIAGNOSTIC_ONLY', branches=['BK', 'BT', 'BTF'],
        updates_per_branch=16, total_update_budget=48, initialize_from=binding['checkpoint'],
        optimizer_options=binding['fresh_optimizer_options'], optimizer_initialization='LOAD_U600_SAVED_STATE',
        rng_initialization='RESTORE_U600_SAVED_PYTHON_NUMPY_TORCH_CUDA_AFTER_PREPARATION',
        samples=samples, draw_plan=plan, output=str(DIAG), teacher_sample=TEACHER_ID,
        teacher_oracle_archive=str(oracle/'COMPLETE_ORACLE_INPUT_AND_OUTPUT.npz'),
        oracle_scientific_review=str(oracle/'SCIENTIFIC_REVIEW.json'),
        teacher_depth_coefficient=1.0, teacher_field_coefficient=1.0, sample_weight=.25,
        teacher_support='teacher first-surface valid AND original source foreground; no Kinect error filtering',
        teacher_field='Saved original B DPSR128/sigma2 base field from stride4 six-source teacher-visible depth normals in original GIVEN cube; not full teacher mesh truth or official GT',
        teacher_missing_0019='MISSING_SOURCE; Kinect retained; teacher depth and field coefficients zero',
        prediction_field='Original B refined indicator, same MSE definition as TH',
        frozen=['vggt.aggregator', 'appearance'], camera_head='absent; original reference K/E',
        acceptance_scope='ONLY_THIS_48_UPDATE_DIAGNOSTIC', formal_training_allowed=False,
        decision_limit='16 updates assess linkage and short adaptation; absence of clear gain does not establish futility of 200 updates',
        restart_policy='Stop on actual failure; no automatic restart or budget extension')
    write_json(path, contract)
    return path


def teacher_target(contract, batch):
    """原camera-Z检查保持；仅非零场监督读取真实indicator，不为depth造场。"""
    need_field = contract.get('teacher_field_coefficient', 1) != 0
    keys = ['K', 'E', 'unit_center', 'unit_scale', 'source_foreground', 'teacher_valid', 'depth']
    if need_field:
        keys.append('indicator_base')
    with np.load(contract['teacher_oracle_archive'], allow_pickle=False) as saved:
        archive = {key: torch.as_tensor(saved[key].copy()) for key in keys}
    for key in ('K', 'E'):
        if not torch.equal(archive[key].unsqueeze(0), batch[key]):
            raise ValueError(f'教师 {key} 与学生输入不一致')
    for key in ('unit_center', 'unit_scale'):
        if not torch.equal(archive[key], batch[key]):
            raise ValueError(f'教师 {key} 与原 B cube 不一致')
    if not torch.equal(archive['source_foreground'].bool().unsqueeze(0), batch['masks'].bool()):
        raise ValueError('教师与学生完整 foreground 不一致')
    valid = archive['teacher_valid'].bool().unsqueeze(0) & batch['masks'].bool()
    depth = archive['depth'].unsqueeze(0).float()
    if not valid.any() or not torch.isfinite(depth[valid]).all():
        raise ValueError('实际教师深度监督为空或非有限')
    target = dict(depth=depth, valid=valid)
    if need_field:
        field = archive['indicator_base'].float()
        if field.shape != (1,128,128,128) or not torch.isfinite(field).all():
            raise ValueError('实际教师场非有限或定义不符')
        target['field'] = field
    return target


def restore_rng(rng):
    """在模型与目标准备后恢复共同 U600 随机状态。"""
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])


def term_gradient(model, loss):
    """只在首个真实更新验证新增项到可训练几何参数的梯度。"""
    named = [(name, p) for name,p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _,p in named], retain_graph=True, allow_unused=True)
    groups = {}
    for (name, _), grad in zip(named, grads):
        if grad is None:
            continue
        group = name.split('.')[0]
        row = groups.setdefault(group, dict(tensors=0, finite=True, squared_norm=0.0))
        row['tensors'] += 1
        row['finite'] &= bool(torch.isfinite(grad).all())
        row['squared_norm'] += float(grad.detach().float().square().sum())
    if not groups or not all(row['finite'] for row in groups.values()) or sum(row['squared_norm'] for row in groups.values()) <= 0:
        raise RuntimeError('新增教师项未产生有限非零几何梯度')
    return groups
