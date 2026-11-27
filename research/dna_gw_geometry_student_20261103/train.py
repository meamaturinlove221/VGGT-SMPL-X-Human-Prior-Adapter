"""按共同 U600 恢复三支，训练期只在原 DNA 源路径添加指定损失。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from research.dna_2dgs_body_depth_student_20261102.student import dna_loss, forward
from research.dna_b_adapter_20261021.bounded_teacher_train import prepare_samples
from research.dna_b_adapter_20261021.paired_train import load_arrays, save_state, write_json
from research.dna_b_adapter_20261021.runtime import geometry_optimizer, load_b_model
from research.dna_b_adapter_20261021.teacher_diagnostic import restore_rng, term_gradient
from research.mentor_human_model.train import geometry_losses, gradient_evidence

from .contract import BRANCHES, OUT, SID, read


def initialize(contract, samples, recovery=None):
    """沿用原参数组，准备原 raster 与 TH 场后恢复检查点随机态。"""
    torch.set_num_threads(4)
    model, saved = load_b_model(recovery or contract['initialize_from'])
    if recovery is None and saved['update'] != 600:
        raise ValueError('恢复权重并非原 U600')
    for name, module in model.named_children():
        if name != 'vggt':
            module.cuda()
    model.vggt.depth_head.cuda()
    optimizer = geometry_optimizer(model, contract['optimizer_options'])
    optimizer.load_state_dict(saved['optimizer'])
    prepared = prepare_samples(model, samples)
    model.train()
    model.vggt.aggregator.eval()
    model.appearance.eval()
    restore_rng(saved['rng'])
    evidence = dict(parent=contract['initialize_from'], parent_update=600,
                    model='Original strict B factory', optimizer='Original saved AdamW',
                    rng='Python NumPy Torch CUDA restored after preparation',
                    frozen=contract['frozen'], camera_head=contract['camera_head'],
                    trainable_modules=sorted({
                        'vggt.depth_head' if name.startswith('vggt.depth_head.') else name.split('.')[0]
                        for name, param in model.named_parameters() if param.requires_grad
                    }))
    resumed = saved['actual_training'] if recovery is not None else None
    if resumed is not None and resumed['actual_updates'] != saved['update']:
        raise ValueError('恢复检查点的有效步数不同')
    return model, optimizer, prepared, evidence, resumed


def teacher_target(contract):
    """两支读取同一冻结物，验证实际 V 与旧固定索引逐值一致。"""
    receipt = read(contract['teacher_target_receipt'])
    if receipt['status'] != 'FROZEN_GW_TARGETS':
        raise ValueError('所选整份教师尚未冻结')
    target = load_arrays(contract['teacher_target_archive'], 'cuda')
    frozen = load_arrays(contract['source_V'], 'cuda')['valid']
    if (not torch.equal(target['valid'], frozen) or not bool(target['normal_valid'].any())
            or bool((target['normal_valid'] & ~frozen).any())):
        raise ValueError('实际教师 V 或 VN 不符')
    return target, receipt


def signed_normal_loss(prediction, teacher):
    """只消费当前可微世界预测，固定 VN 上直接使用有符号余弦。"""
    valid = teacher['normal_valid']
    student = normalize(prediction['world_normal'][valid], dim=-1, eps=1e-6)
    target = normalize(teacher['world_normal'][valid], dim=-1, eps=1e-6)
    return (1 - (student * target).sum(-1)).mean()


def dna_terms(prediction, original_target, batch, teacher=None, normal_weight=0.):
    """原 Kinect 均值与固定 V 的独立新均值相加，缺测不进入算式。"""
    loss, parts = dna_loss(prediction, original_target, batch)
    if teacher is not None:
        valid = teacher['valid']
        scale = batch['unit_scale'].reshape(-1, 1, 1, 1).expand_as(valid)
        parts['teacher_depth'] = (
            (prediction['depth'][valid] - teacher['depth'][valid]).abs() / scale[valid]).mean()
        loss = loss + parts['teacher_depth']
        if normal_weight:
            parts['teacher_normal'] = signed_normal_loss(prediction, teacher)
            loss = loss + normal_weight * parts['teacher_normal']
    return loss, parts


def check_first_batch(contract):
    """唯一真实首样本梯度检查，保存后丢弃零更新实例。"""
    path = OUT / 'FIRST_BATCH_NORMAL_GRADIENT.json'
    if path.exists():
        raise FileExistsError(path)
    target, receipt = teacher_target(contract)
    row = next(item for item in contract['samples'] if item['id'] == SID)
    model, optimizer, prepared, evidence, _ = initialize(contract, [row])
    write_json(path, dict(status='FORWARD_STARTED', **evidence, optimizer_steps=0,
                          teacher_mesh=receipt['mesh'], network_forwards_started=1))
    prediction, original_target, batch = forward(model, prepared[SID])
    total, parts = dna_terms(prediction, original_target, batch, target, .05)
    normal = parts['teacher_normal']
    groups = term_gradient(model, .25 * .05 * normal)
    normal_grad = groups.get('normal_head')
    if not normal_grad or not normal_grad['finite'] or normal_grad['squared_norm'] <= 0:
        raise RuntimeError('真实正常权重的有符号损失没有到达原 normal_head')
    write_json(path, dict(
        status='PASS', **evidence, teacher_mesh=receipt['mesh'],
        optimizer_steps=0, network_forwards=1, discarded_instance=True,
        depth_pixels=int(target['valid'].sum()), normal_pixels=int(target['normal_valid'].sum()),
        loss={**{key: float(value.detach()) for key, value in parts.items()},
              'sample_total_weighted': float((.25 * total).detach())},
        gradient_of_normal_term_scaled_by_025_and_005=groups,
    ))
    print(path, flush=True)


def train(contract, branch, recover=False):
    """每个分支从独立实例运行固定 200 步并保存 U0/U100/U200。"""
    depth_weight, normal_weight = BRANCHES[branch]
    normal_check = read(OUT / 'FIRST_BATCH_NORMAL_GRADIENT.json')
    if normal_check['status'] != 'PASS':
        raise ValueError('缺实际 normal_head 梯度回执')
    if depth_weight:
        target, receipt = teacher_target(contract)
    else:
        target, receipt = None, read(contract['teacher_target_receipt'])
    if normal_check['teacher_mesh'] != receipt['mesh']:
        raise ValueError('首样本检查和正式训练的教师不同')
    folder = OUT / branch
    recovery = folder / 'checkpoint_failure.pt' if recover else None
    if recovery is None:
        folder.mkdir(exist_ok=False)
    model, optimizer, prepared, evidence, resumed = initialize(contract, contract['samples'], recovery)
    depth_pixels = int(target['valid'].sum()) if depth_weight else 0
    normal_pixels = int(target['normal_valid'].sum()) if normal_weight else 0
    if resumed is None:
        write_json(folder / 'SAME_INITIAL_STATE_EVIDENCE.json', evidence)
        write_json(folder / 'DRAW_PLAN.json', contract['draw_plan'])
        record = dict(status='RUNNING', branch=branch, **evidence, actual_updates=0,
                      physical_updates=0, geometry_updates=0, network_forwards=0,
                      aggregator_forwards=0, appearance_updates=0, teacher_exposures=0,
                      teacher_normal_exposures=0, exposures={}, teacher_depth_weight=depth_weight,
                      teacher_normal_weight=normal_weight, teacher_field_weight=0,
                      teacher_depth_pixels=depth_pixels, teacher_normal_pixels=normal_pixels,
                      teacher_mesh=receipt['mesh'], source_contract=str(OUT / 'TRAINING_CONTRACT.json'))
        save_state(model, optimizer, folder / 'checkpoint_u0000.pt', 0, record)
    else:
        # 已提交的步数来自完整模型/AdamW/RNG检查点，续接下一 draw。
        record = resumed
        lines = (folder / 'updates.jsonl').read_text(encoding='utf8').splitlines()
        committed = record['actual_updates']
        if (record['status'] != 'STOPPED_REAL_ERROR' or record['branch'] != branch
                or not 0 < committed < 200 or len(lines) != committed
                or json.loads(lines[-1])['update'] != committed
                or read(folder / 'DRAW_PLAN.json') != contract['draw_plan']):
            raise ValueError('错误前完整检查点与已提交 draw 不符')
        write_json(folder / 'RECOVERY.json', dict(
            checkpoint=str(recovery), completed_steps=committed, first_remaining_step=committed + 1,
            previous_error=record.pop('error'), additional_check_forwards=0, repeated_steps=0))
        record.update(status='RUNNING', recovered_from_update=committed)
    write_json(folder / 'RUN.json', record)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with (folder / 'updates.jsonl').open('a' if recover else 'w', encoding='utf8') as log:
            for draw in contract['draw_plan'][record['actual_updates']:]:
                rows = []
                step_start = time.perf_counter()
                for domain in ('DNA', 'TH'):
                    for sid in draw[domain]:
                        prediction, original_target, batch = forward(model, prepared[sid])
                        record['network_forwards'] += 1
                        use_depth = sid == SID and depth_weight > 0
                        use_normal = sid == SID and normal_weight > 0
                        if domain == 'TH':
                            loss, parts = geometry_losses(
                                prediction, original_target, batch, prepared[sid]['field'].cuda())
                        else:
                            loss, parts = dna_terms(
                                prediction, original_target, batch, target if use_depth else None,
                                normal_weight if use_normal else 0.)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f'真实非有限损失 {sid}')
                        (.25 * loss).backward()
                        rows.append(dict(
                            id=sid, domain=domain, weight=.25,
                            teacher_depth_valid=depth_pixels if use_depth else 0,
                            teacher_normal_valid=normal_pixels if use_normal else 0,
                            loss={key: float(value.detach()) for key, value in parts.items()},
                            weighted_total=float((.25 * loss).detach())))
                        del prediction, original_target, batch, loss, parts
                update = draw['update']
                if update == 1:
                    write_json(folder / 'FIRST_UPDATE_GRADIENT_MODULES.json', gradient_evidence(model))
                if any(param.grad is not None and not bool(torch.isfinite(param.grad).all())
                       for param in model.parameters()):
                    raise RuntimeError('真实非有限梯度')
                optimizer.step()
                record.update(actual_updates=update, physical_updates=update, geometry_updates=update)
                for row in rows:
                    sid = row['id']
                    record['exposures'][sid] = record['exposures'].get(sid, 0) + 1
                    record['teacher_exposures'] += int(row['teacher_depth_valid'] > 0)
                    record['teacher_normal_exposures'] += int(row['teacher_normal_valid'] > 0)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                report = dict(update=update, draws=rows, seconds=time.perf_counter() - step_start)
                log.write(json.dumps(report) + '\n')
                log.flush()
                if update == 200:
                    record['status'] = 'COMPLETE_EXPLORATORY_GW200'
                if update in contract['checkpoints']:
                    save_state(model, optimizer, folder / f'checkpoint_u{update:04d}.pt', update, record)
                write_json(folder / 'RUN.json', record)
    except Exception as error:
        record.update(status='STOPPED_REAL_ERROR', error=repr(error))
        error_file = 'checkpoint_recovery_failure.pt' if recover else 'checkpoint_failure.pt'
        save_state(model, optimizer, folder / error_file, record['actual_updates'], record)
        raise
    finally:
        record.update(loop_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(folder / 'RUN.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', choices=['check', *BRANCHES])
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    contract = read(OUT / 'TRAINING_CONTRACT.json')
    if args.branch == 'check':
        check_first_batch(contract)
    else:
        train(contract, args.branch, args.recover)
