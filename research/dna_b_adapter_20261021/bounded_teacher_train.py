"""复用原B工厂：保留旧48步诊断，新增明确合同下的两支200更新。"""
import argparse
import json
import time
from pathlib import Path
import torch
from research.dna_b_adapter_20261021.runtime import load_b_model, geometry_optimizer
from research.dna_b_adapter_20261021.paired_train import load_arrays, save_state, write_json
from research.dna_b_adapter_20261021.teacher_diagnostic import prepare_contract, teacher_target, restore_rng, term_gradient
from research.dna_b_adapter_20261021.teacher_supervision import SCOPE, approved_target
from research.mentor_human_model.train import geometry_losses, gradient_evidence, move


def prepare_samples(model, samples):
    """沿用旧训练的 raster 与四个 TH 原场目标，准备阶段不更新参数。"""
    prepared = {}
    for item in samples:
        batch = load_arrays(item['input_npz']); batch['smpl_faces'] = batch['smpl_faces'].long()
        target = load_arrays(item['target_npz']); gpu = move(batch, 'cuda')
        with torch.no_grad():
            raster = move(model.prepare_raster(gpu), 'cpu')
            field = None
            if item['domain'] == 'TH':
                field = model.surface.target(target['scan_points'].cuda(), target['scan_normals'].cuda(),
                    gpu['unit_center'].reshape(-1,3), gpu['unit_scale'].reshape(-1)).detach().cpu()
        prepared[item['id']] = dict(item=item, batch=batch, target=target, raster=raster, field=field)
        del gpu
    return prepared


def train(contract_path, branch):
    """严格恢复同权重、Adam和RNG，只执行已冻结的诊断或正式有限分支。"""
    contract = json.loads(Path(contract_path).read_text(encoding='utf8'))
    formal = contract['scope'] == SCOPE
    if formal:
        if (contract['updates_per_branch'] != 200 or contract['total_update_budget'] != 400
                or branch not in ('K', 'T') or contract['formal_training_allowed'] is not True):
            raise ValueError('只允许固定K/T各200更新，不自动加入field分支')
    elif contract['updates_per_branch'] != 16 or contract['total_update_budget'] != 48 or branch not in ('BK','BT','BTF'):
        raise ValueError('只允许固定三组各 16 步诊断')
    baseline = branch in ('BK', 'K')
    endpoint = contract['updates_per_branch']
    completed_status = 'COMPLETE_PAIRED_200' if formal else 'COMPLETE_DIAGNOSTIC_16'
    out = Path(contract['output'])/branch
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, saved = load_b_model(contract['initialize_from'])
    for name,module in model.named_children():
        if name != 'vggt': module.cuda()
    model.vggt.depth_head.cuda()
    optimizer = geometry_optimizer(model, contract['optimizer_options'])
    optimizer.load_state_dict(saved['optimizer'])
    prepared = prepare_samples(model, contract['samples'])
    teachers = ({sid: approved_target(entry, prepared[sid]['batch']) for sid, entry in contract['teachers'].items()}
                if formal else {contract['teacher_sample']: teacher_target(contract, prepared[contract['teacher_sample']]['batch'])})
    record = dict(status='RUNNING', branch=branch, actual_updates=0, physical_updates=0,
        geometry_updates=0, network_forwards=0, aggregator_forwards=0, appearance_updates=0,
        parent=contract['initialize_from'], parent_update=int(saved['update']),
        optimizer='U600 saved AdamW state; original lr and moments retained',
        source_contract=str(Path(contract_path).resolve()), exposures={},
        teacher_visible_foreground={sid:int(target['valid'].sum()) for sid,target in teachers.items()},
        teacher_depth_coefficient=0 if baseline else 1,
        teacher_field_coefficient=1 if branch=='BTF' else 0,
        scope=contract['scope'], formal_training_allowed=formal)
    write_json(out/'DRAW_PLAN.json',contract['draw_plan'])
    write_json(out/'RUN.json',record)
    model.train(); model.vggt.aggregator.eval(); model.appearance.eval()
    restore_rng(saved['rng']); del saved
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    teacher_gradient_recorded = False
    try:
        with (out/'updates.jsonl').open('w',encoding='utf8') as log:
            for draw in contract['draw_plan']:
                optimizer.zero_grad(set_to_none=True); rows=[]; step_start=time.perf_counter()
                for domain in ('DNA','TH'):
                    for sid in draw[domain]:
                        entry=prepared[sid]; batch=move(entry['batch'],'cuda'); target=move(entry['target'],'cuda')
                        raster=move(entry['raster'],'cuda')
                        packet=torch.load(entry['item']['snapshots'],map_location='cpu',weights_only=False,mmap=True)
                        record['network_forwards']+=1
                        is_teacher=sid in teachers
                        if domain=='TH' or (is_teacher and not formal):
                            pred=model(batch,snapshots=packet,raster=raster)
                        else:
                            pred=model.forward_source_depth(batch,snapshots=packet,raster=raster)
                        if domain=='TH':
                            loss,parts=geometry_losses(pred,target,batch,entry['field'].cuda())
                        else:
                            valid=target['depth_valid'].bool()
                            loss=((pred['depth']-target['depth']).abs()/batch['unit_scale'].reshape(-1,1,1,1))[valid].mean()
                            parts={'kinect_depth':loss}
                            if is_teacher and not baseline:
                                teacher=teachers[sid]
                                support=teacher['valid'].cuda()
                                td=((pred['depth']-teacher['depth'].cuda()).abs()/batch['unit_scale'].reshape(-1,1,1,1))[support].mean()
                                parts['teacher_depth']=td; loss=loss+td
                                if not teacher_gradient_recorded:
                                    write_json(out/'TEACHER_DEPTH_GRADIENT.json',term_gradient(model,td))
                                if branch=='BTF':
                                    tf=(pred['indicator']-teacher['field'].cuda()).square().mean()
                                    parts['teacher_field']=tf; loss=loss+tf
                                    if not teacher_gradient_recorded:
                                        write_json(out/'TEACHER_FIELD_GRADIENT.json',term_gradient(model,tf))
                                teacher_gradient_recorded = True
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f'真实非有限 loss: {sid}')
                        (loss*.25).backward()
                        rows.append(dict(id=sid,domain=domain,weight=.25,
                            loss={key:float(value.detach()) for key,value in parts.items()}))
                        del pred,loss,parts,packet,batch,target,raster
                        if is_teacher and not baseline:
                            del td,support
                            if branch=='BTF': del tf
                if draw['update']==1:
                    write_json(out/'FIRST_REAL_UPDATE_GRADIENT.json',gradient_evidence(model))
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError('真实梯度非有限')
                optimizer.step()
                record.update(actual_updates=draw['update'],physical_updates=draw['update'],geometry_updates=draw['update'])
                for row in rows:
                    record['exposures'][row['id']]=record['exposures'].get(row['id'],0)+1
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.synchronize()
                report=dict(update=draw['update'],draws=rows,seconds=time.perf_counter()-step_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(report)+'\n'); log.flush(); print(json.dumps(report),flush=True)
                write_json(out/'RUN.json',record)
                if draw['update'] in (contract['checkpoint_updates'] if formal else [8,16]):
                    if draw['update']==endpoint: record['status']=completed_status
                    save_state(model,optimizer,out/f"checkpoint_u{draw['update']:04d}.pt",draw['update'],record)
        record['status']=completed_status
    except Exception as error:
        record.update(status='STOPPED_REAL_ERROR',error=repr(error))
        save_state(model,optimizer,out/'checkpoint_failure.pt',record['actual_updates'],record)
        raise
    finally:
        record['loop_seconds']=time.perf_counter()-started
        write_json(out/'RUN.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--contract')
    parser.add_argument('--branch',choices=('BK','BT','BTF','K','T'))
    args=parser.parse_args()
    if args.prepare:
        print(prepare_contract())
    else:
        train(args.contract,args.branch)
