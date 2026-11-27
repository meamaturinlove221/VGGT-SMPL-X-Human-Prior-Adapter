"""复用 0019 的逐步训练循环，只把新增 GW 监督绑定到每步首 DNA 槽。"""
import json
import time

import torch
from research.dna_gw_geometry_student_20261103.train import (
    BRANCHES, dna_terms, forward, geometry_losses, gradient_evidence,
    initialize, read, save_state, teacher_target, write_json,
)


def verify_u0(output, branch):
    """在实际首步前逐值核对该实例的 U0、AdamW 和随机态。"""
    import verify_u0 as original
    original.OUT = output
    original.main([branch])


def train(contract, branch, output, recover=False):
    """每个分支从独立实例运行固定 200 步并保存 U0/U100/U200。"""
    depth_weight, normal_weight = BRANCHES[branch]
    normal_check = read(output / 'FIRST_BATCH_NORMAL_GRADIENT.json')
    if normal_check['status'] != 'PASS':
        raise ValueError('缺实际 normal_head 梯度回执')
    if depth_weight:
        target, receipt = teacher_target(contract)
    else:
        target, receipt = None, read(contract['teacher_target_receipt'])
    if normal_check['teacher_mesh'] != receipt['mesh']:
        raise ValueError('首样本检查和正式训练的教师不同')
    folder = output / branch
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
                      teacher_mesh=receipt['mesh'], source_contract=str(output / 'TRAINING_CONTRACT.json'))
        save_state(model, optimizer, folder / 'checkpoint_u0000.pt', 0, record)
        verify_u0(output, branch)
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
                    for slot, sid in enumerate(draw[domain]):
                        prediction, original_target, batch = forward(model, prepared[sid])
                        record['network_forwards'] += 1
                        use_depth = (domain == 'DNA' and slot == 0) and depth_weight > 0
                        use_normal = (domain == 'DNA' and slot == 0) and normal_weight > 0
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
