"""复用原B工厂、准备和loss的单0019有限训练入口。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from research.dna_b_adapter_20261021.runtime import load_b_model, geometry_optimizer
from research.dna_b_adapter_20261021.bounded_teacher_train import prepare_samples
from research.dna_b_adapter_20261021.paired_train import load_arrays, save_state, write_json
from research.dna_b_adapter_20261021.teacher_supervision import approved_target
from research.dna_b_adapter_20261021.teacher_diagnostic import restore_rng, term_gradient
from research.mentor_human_model.train import geometry_losses, move
from research.dna_2dgs_body_depth_student_20261102.contract import OUT, SID, SCOPE, prepare, read


def dna_loss(pred, target, batch, teacher=None):
    """与旧正式入口相同的两个独立均值，新增项只取固定实际支持。"""
    scale = batch['unit_scale'].reshape(-1,1,1,1)
    k = ((pred['depth']-target['depth']).abs()/scale)[target['depth_valid'].bool()].mean()
    parts = dict(kinect_depth=k)
    if teacher is not None:
        parts['teacher_depth'] = ((pred['depth']-teacher['depth'].cuda()).abs()/scale)[teacher['valid'].cuda()].mean()
    return sum(parts.values()), parts


def initialize(contract, samples):
    """准备后再恢复原U600 RNG，检查实例与正式实例彼此独立。"""
    torch.set_num_threads(4)
    model, saved = load_b_model(contract['initialize_from'])
    for name, module in model.named_children():
        if name != 'vggt': module.cuda()
    model.vggt.depth_head.cuda()
    optimizer = geometry_optimizer(model, contract['optimizer_options'])
    optimizer.load_state_dict(saved['optimizer'])
    prepared = prepare_samples(model, samples)
    teacher = approved_target(contract['teachers'][SID], prepared[SID]['batch'])
    model.train(); model.vggt.aggregator.eval(); model.appearance.eval()
    restore_rng(saved['rng'])
    evidence = dict(parent=contract['initialize_from'], parent_update=int(saved['update']),
                    model='Strict original load_b_model state loading', optimizer='Original saved AdamW state loaded',
                    RNG='Original saved Python NumPy Torch CUDA restored after preparation',
                    check_instance_reused=False)
    return model, optimizer, prepared, teacher, evidence


def forward(model, entry):
    """保持原缓存与DNA source-depth、TH完整几何路径。"""
    batch=move(entry['batch'],'cuda'); target=move(entry['target'],'cuda'); raster=move(entry['raster'],'cuda')
    packet=torch.load(entry['item']['snapshots'],map_location='cpu',weights_only=False,mmap=True)
    pred=(model(batch,snapshots=packet,raster=raster) if entry['item']['domain']=='TH'
          else model.forward_source_depth(batch,snapshots=packet,raster=raster))
    return pred, target, batch


def check(contract):
    """唯一真实batch检查实际公式与新增梯度；零更新后丢弃实例。"""
    folder=OUT/'LOSS_MASK_ZERO_UPDATE'; folder.mkdir(exist_ok=False)
    started=time.perf_counter()
    items=[item for item in contract['samples'] if item['id']==SID]
    model,optimizer,prepared,teacher,evidence=initialize(contract,items)
    pred,target,batch=forward(model,prepared[SID])
    loss,parts=dna_loss(pred,target,batch,teacher)
    p=pred['depth'].detach().cpu().numpy(); k=target['depth'].cpu().numpy()
    t=teacher['depth'].numpy(); kval=target['depth_valid'].bool().cpu().numpy(); valid=teacher['valid'].numpy()
    scale=batch['unit_scale'].detach().cpu().numpy()
    kn=float((np.abs(p.astype(np.float64)-k)/scale.reshape(-1,1,1,1))[kval].mean())
    tn=float((np.abs(p.astype(np.float64)-t)/scale.reshape(-1,1,1,1))[valid].mean())
    delta=max(abs(float(parts['kinect_depth'])-kn),abs(float(parts['teacher_depth'])-tn),abs(float(loss)-kn-tn))
    if not np.isfinite(delta) or delta>1e-6: raise RuntimeError('实际独立均值复算不符')
    gradient=term_gradient(model,parts['teacher_depth']); (loss*.25).backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError('实际合并梯度非有限')
    np.savez_compressed(folder/'ACTUAL_LOSS_RECOMPUTABLE.npz',pred=p,kinect=k,teacher=t,kinect_valid=kval,
                        teacher_valid=valid,unit_scale=scale)
    write_json(folder/'RECEIPT.json',dict(status='PASS',**evidence,optimizer_steps=0,network_forwards=1,
        kinect_valid=int(kval.sum()),teacher_valid=int(valid.sum()),K_float64=kn,T_float64=tn,
        actual_parts={key:float(value.detach()) for key,value in parts.items()},max_absolute_difference=delta,
        teacher_gradient=gradient,teacher_coefficient=1,outer_weight=.25,normal=0,field=0,
        seconds=time.perf_counter()-started,discarded_instance=True))


def train(contract):
    """唯一T200端点，不改变原draw，不自动重试或延长。"""
    if contract['scope']!=SCOPE or read(OUT/'LOSS_MASK_ZERO_UPDATE/RECEIPT.json')['status']!='PASS':
        raise ValueError('单0019合同与一次实际检查不符')
    folder=OUT/'T';folder.mkdir(exist_ok=False)
    model,optimizer,prepared,teacher,evidence=initialize(contract,contract['samples'])
    write_json(folder/'SAME_INITIAL_STATE_EVIDENCE.json',evidence)
    write_json(folder/'DRAW_PLAN.json',contract['draw_plan'])
    record=dict(status='RUNNING',**evidence,actual_updates=0,physical_updates=0,geometry_updates=0,
        network_forwards=0,aggregator_forwards=0,appearance_updates=0,exposures={},teacher_exposures=0,
        teacher_visible_foreground={SID:int(teacher['valid'].sum())},source_contract=str(OUT/'TRAINING_CONTRACT.json'),
        teacher_depth_coefficient=1,teacher_normal_coefficient=0,teacher_field_coefficient=0)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        with (folder/'updates.jsonl').open('w',encoding='utf8') as log:
            for draw in contract['draw_plan']:
                optimizer.zero_grad(set_to_none=True); rows=[]; step_start=time.perf_counter()
                for domain in ('DNA','TH'):
                    for sid in draw[domain]:
                        entry=prepared[sid];pred,target,batch=forward(model,entry);record['network_forwards']+=1
                        if domain=='TH': loss,parts=geometry_losses(pred,target,batch,entry['field'].cuda())
                        else: loss,parts=dna_loss(pred,target,batch,teacher if sid==SID else None)
                        if not bool(torch.isfinite(loss)): raise RuntimeError('真实非有限loss '+sid)
                        (loss*.25).backward()
                        rows.append(dict(id=sid,domain=domain,weight=.25,
                            teacher_valid=int(teacher['valid'].sum()) if sid==SID else 0,
                            loss={key:float(value.detach()) for key,value in parts.items()}))
                        del pred,target,batch,loss,parts
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError('真实非有限梯度')
                optimizer.step();record.update(actual_updates=draw['update'],physical_updates=draw['update'],geometry_updates=draw['update'])
                for row in rows: record['exposures'][row['id']]=record['exposures'].get(row['id'],0)+1
                record['teacher_exposures']+=int(SID in draw['DNA'])
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.synchronize()
                report=dict(update=draw['update'],draws=rows,seconds=time.perf_counter()-step_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(report)+'\n');log.flush();print(json.dumps(report),flush=True)
                write_json(folder/'RUN.json',record)
                if draw['update'] in contract['checkpoint_updates']:
                    if draw['update']==200: record['status']='COMPLETE_EXPLORATORY_T200'
                    save_state(model,optimizer,folder/f"checkpoint_u{draw['update']:04d}.pt",draw['update'],record)
    except Exception as error:
        record.update(status='STOPPED_REAL_ERROR',error=repr(error))
        save_state(model,optimizer,folder/'checkpoint_failure.pt',record['actual_updates'],record)
        raise
    finally:
        record.update(loop_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(folder/'RUN.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','check','train']);args=parser.parse_args()
    if args.mode=='prepare': print(prepare())
    elif args.mode=='check': check(read(OUT/'TRAINING_CONTRACT.json'))
    else: train(read(OUT/'TRAINING_CONTRACT.json'))
