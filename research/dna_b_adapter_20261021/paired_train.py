"""B原几何工厂的有限Kinect适配，DNA与TH域各占一半。"""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from research.dna_b_adapter_20261021.runtime import load_b_model, geometry_optimizer
from research.mentor_human_model.train import geometry_losses, gradient_evidence, move


def write_json(path, value):
    """保存实际收据，不将计划数写成完成数。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def load_arrays(path, device='cpu'):
    """读取真实数组，不改变单位、像素和有效支持。"""
    with np.load(path, allow_pickle=False) as z:
        return {key: torch.as_tensor(z[key].copy(), device=device) for key in z.files}


def save_state(model, optimizer, path, update, record):
    """保存所有可恢复几何外观参数、优化器及随机状态，冻结主干仅引用。"""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith('vggt.aggregator.')}
    torch.save(dict(config=model.config, model=state, optimizer=optimizer.state_dict(),
                    update=update, actual_training=record,
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                             torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), path)


def balanced_plan(samples, seed, updates):
    """按身份循环随机队列，各身份帧轮换，所有困难帧不按误差排除。"""
    rng = random.Random(seed)
    dna, th = {}, []
    for item in samples:
        if item['domain'] == 'DNA':
            dna.setdefault(item['id'].split('/')[1], []).append(item['id'])
        else:
            th.append(item['id'])
    ids = sorted(dna)
    order = []
    while len(order) < updates * 2:
        cycle = ids.copy()
        rng.shuffle(cycle)
        order.extend(cycle)
    counters = dict.fromkeys(ids, 0)
    plan = []
    for step in range(updates):
        draws = []
        for identity in order[step*2:step*2+2]:
            frames = dna[identity]
            draws.append(frames[counters[identity] % len(frames)])
            counters[identity] += 1
        plan.append(dict(update=step+1, DNA=draws,
                         TH=[th[(step*2+i) % len(th)] for i in range(2)]))
    return plan


def train(contract_path):
    """执行预先登记的一支有限训练；真实错误保留现场，不追加恢复步数。"""
    contract = json.loads(Path(contract_path).read_text(encoding='utf8'))
    out = Path(contract['output']); out.mkdir(parents=True, exist_ok=True)
    seed, updates = contract['seed'], contract['updates']
    if updates != 600:
        raise ValueError('本轮固定600几何更新，不接受隐式预算扩展')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    samples = json.loads(Path(contract['manifest']).read_text(encoding='utf8'))['train']
    plan = balanced_plan(samples, seed, updates)
    write_json(out/'DRAW_PLAN.json', plan)
    model, saved = load_b_model(contract['initialize_from'])
    del saved
    for name, module in model.named_children():
        if name != 'vggt':
            module.cuda()
    model.vggt.depth_head.cuda()
    optimizer = geometry_optimizer(model, contract['optimizer'])
    prepared = {}
    for item in samples:
        batch = load_arrays(item['input_npz'])
        batch['smpl_faces'] = batch['smpl_faces'].long()
        gpu = move(batch, 'cuda')
        raster = model.prepare_raster(gpu)
        target = load_arrays(item['target_npz'])
        field = None
        if item['domain'] == 'TH':
            with torch.no_grad():
                field = model.surface.target(target['scan_points'].cuda(), target['scan_normals'].cuda(),
                    gpu['unit_center'].reshape(-1,3), gpu['unit_scale'].reshape(-1)).detach().cpu()
        prepared[item['id']] = dict(item=item, batch=batch, target=target,
            raster=move(raster, 'cpu'), field=field)
        del gpu
    record = dict(status='RUNNING', actual_updates=0, physical_updates=0,
                  network_forwards=0, aggregator_forwards=0, geometry_updates=0,
                  appearance_updates=0, parent=contract['initialize_from'],
                  source_contract=str(Path(contract_path).resolve()), exposures={},
                  domain_weight=dict(DNA=.5, TH=.5), camera_head='absent',
                  DNA_normal_loss=0, DNA_world_loss=0, RNG_seed=seed)
    save_state(model, optimizer, out/'checkpoint_u0000.pt', 0, record)
    model.train(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    try:
        with (out/'updates.jsonl').open('w', encoding='utf8') as log:
            for draw in plan:
                step_start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                rows=[]
                for domain in ('DNA', 'TH'):
                    for sid in draw[domain]:
                        entry=prepared[sid]; item=entry['item']
                        batch=move(entry['batch'],'cuda'); target=move(entry['target'],'cuda')
                        packet=torch.load(item['snapshots'],map_location='cpu',weights_only=False,mmap=True)
                        raster=move(entry['raster'],'cuda')
                        record['network_forwards'] += 1
                        if domain=='DNA':
                            pred=model.forward_source_depth(batch,snapshots=packet,raster=raster)
                            valid=target['depth_valid'].bool()
                            loss=((pred['depth']-target['depth']).abs()/batch['unit_scale'].reshape(-1,1,1,1))[valid].mean()
                            parts={'depth':loss}
                        else:
                            pred=model(batch,snapshots=packet,raster=raster)
                            loss,parts=geometry_losses(pred,target,batch,entry['field'].cuda())
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f'真实非有限loss: {sid}')
                        (loss*.25).backward()
                        rows.append(dict(id=sid,domain=domain,weight=.25,
                            valid_count=int(target['depth_valid'].sum()),
                            loss={key:float(value.detach()) for key,value in parts.items()}))
                        del pred,loss,parts,packet,batch,target,raster
                if draw['update']==1:
                    evidence=gradient_evidence(model)
                    write_json(out/'FIRST_REAL_UPDATE_GRADIENT.json',evidence)
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError('真实梯度非有限')
                optimizer.step()
                record['actual_updates']=draw['update']; record['physical_updates']=draw['update']
                record['geometry_updates']=draw['update']
                for row in rows:
                    record['exposures'][row['id']]=record['exposures'].get(row['id'],0)+1
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.synchronize()
                row=dict(update=draw['update'],draws=rows,step_seconds=time.perf_counter()-step_start,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row)+'\n');log.flush()
                print(json.dumps(row),flush=True)
                write_json(out/'RUN.json',record)
                if draw['update'] in (200,600):
                    if draw['update']==600:
                        record['status']='COMPLETE_FIXED_600'
                    save_state(model,optimizer,out/f"checkpoint_u{draw['update']:04d}.pt",draw['update'],record)
        record['status']='COMPLETE_FIXED_600'
    except Exception as error:
        record['status']='STOPPED_REAL_ERROR';record['error']=repr(error)
        save_state(model,optimizer,out/'checkpoint_failure.pt',record['actual_updates'],record)
        raise
    finally:
        record['loop_seconds']=time.perf_counter()-started
        write_json(out/'RUN.json',record)
