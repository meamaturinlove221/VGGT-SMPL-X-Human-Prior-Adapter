"""合法四训练身份六图实学；scan仅进入监督，最多2000个optimizer updates。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.mentor_human_model.model.network import MentorHumanModel, backproject


def json_write(path, value):
    """保存可读真实运行记录，不写无消费者摘要。"""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def move(value,device):
    """只迁移张量，不改变源输入数值。"""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value,dict):
        return {k:move(v,device) for k,v in value.items()}
    return value


def masked_mean(values,valid):
    """所有实际有效源像素等权平均，分母为有效像素数量。"""
    return values[valid].mean()


def geometry_losses(prediction,target,batch,target_field):
    """深度/点按源定义cube边长归一化，normal按有效像素，χ按全部体素。"""
    scale = batch['unit_scale'].reshape(-1,1,1,1)
    valid = target['depth_valid'].bool()
    normal_valid = target['normal_valid'].bool() & prediction['normal_valid']
    depth = masked_mean((prediction['depth']-target['depth']).abs()/scale,valid)
    target_world = backproject(target['depth'],batch['K'],batch['E'])
    point = masked_mean((prediction['world_points']-target_world).abs().mean(-1)/scale,valid)
    normal = masked_mean(1-(prediction['world_normal']*target['world_normal']).sum(-1).clamp(-1,1),normal_valid)
    field = (prediction['indicator']-target_field).square().mean()
    total = depth + point + .1*normal + field
    return total,dict(depth=depth,point=point,normal=normal,field=field)


def save_checkpoint(model,optimizer,path,update,record,appearance=None):
    """保存全部训练参数与优化器；冻结主干按真实原始权重路径加载。"""
    state = {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('vggt.aggregator.')}
    torch.save(dict(config=model.config,model=state,optimizer=optimizer.state_dict(),update=update,
                    actual_training=record,appearance=appearance),path)


def gradient_evidence(model):
    """首个真实loss回传后记录每个实际可训练模块，缺失梯度就不声称其已训练。"""
    modules = dict(depth=model.vggt.depth_head,sparse=model.sparse,vertex=model.vertex_features,
                   P3D_bridge=model.prior_bridge,fusion=model.fusion,normal=model.normal_head,surface=model.surface)
    result = {}
    for name,module in modules.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        result[name] = dict(tensors=len(grads),finite=all(bool(torch.isfinite(g).all()) for g in grads),
            norm=float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.)
    return result


def appearance_stage(model, prepared, updates, output, execution, started):
    """冻结学到的衣服几何，以四身份合法源RGB训练同一个Gaussian共享head。"""
    from research.mentor_human_model.model.surface import export_mesh
    from research.mentor_human_model.model.gaussian import SurfaceGaussians
    model.eval().requires_grad_(False)
    model.appearance.train().requires_grad_(True)
    scenes = []
    for record in prepared:
        batch = record['batch']
        with torch.no_grad():
            prediction = model(batch,snapshots=record['snapshots'],raster=record['raster'])
            v,f = export_mesh(prediction['indicator'],batch['unit_center'].cpu().numpy(),
                              batch['unit_scale'].cpu().numpy(),output/f"{record['id'].split('/')[1]}_appearance_mesh.ply")
            gs = SurfaceGaussians.from_mesh(v,f,max_gaussians=12000).cuda().requires_grad_(False)
            gs.initialize_colors(batch['images'][0].permute(0,2,3,1),batch['masks'][0],
                                 batch['E'][0],batch['K'][0],depths=prediction['depth'][0])
        h,w = batch['images'].shape[-2:]
        height,width = h//4,w//4
        Ks = batch['K'][0].clone()
        Ks[:,0] *= width/w
        Ks[:,1] *= height/h
        Ks[:,0,2] += (width/w-1)*.5
        Ks[:,1,2] += (height/h-1)*.5
        images = torch.nn.functional.interpolate(batch['images'][0],size=(height,width),mode='area').permute(0,2,3,1)
        masks = torch.nn.functional.interpolate(batch['masks'][0,:,None].float(),size=(height,width),mode='area')[:,0]
        scenes.append(dict(record=record,gs=gs,K=Ks,images=images,masks=masks,height=height,width=width))
    optimizer = torch.optim.AdamW(model.appearance.parameters(),lr=1e-3)
    execution['appearance_updates'] = 0
    with (output/'appearance_updates.jsonl').open('a',encoding='utf-8') as log:
        for update in range(1,updates+1):
            scene = scenes[(update-1)%len(scenes)]
            record,batch,gs = scene['record'],scene['record']['batch'],scene['gs']
            # 每次只渲染一个合法源视角，六图RGB仍全部参与source条件颜色。
            view = ((update-1)//len(scenes))%6
            optimizer.zero_grad(set_to_none=True)
            values = model.appearance(gs,batch['unit_center'],batch['unit_scale'])
            rendered = gs.render(batch['E'][0,view:view+1],scene['K'][view:view+1],scene['width'],scene['height'],appearance=values)
            rgb = (rendered['rgb'][0]-scene['images'][view]*scene['masks'][view,:, :,None]).abs().mean()
            mask = (rendered['mask'][0]-scene['masks'][view]).abs().mean()
            total = rgb+.1*mask
            if not bool(torch.isfinite(total)):
                raise RuntimeError('共享Gaussian外观目标非有限，保存停止')
            total.backward()
            if update==1:
                grads = [p.grad for p in model.appearance.parameters() if p.grad is not None]
                evidence = dict(tensors=len(grads),finite=all(bool(torch.isfinite(g).all()) for g in grads),
                    norm=float(torch.sqrt(sum(g.square().sum() for g in grads))))
                json_write(output/'appearance_first_gradient.json',evidence)
                if not evidence['finite'] or evidence['norm']==0:
                    raise RuntimeError('共享Gaussian外观head没有有效梯度')
            optimizer.step()
            execution['appearance_updates']=update
            execution['optimizer_updates']+=1
            if 'actual_updates' in execution:
                execution['actual_updates']=execution['optimizer_updates']
            row = dict(appearance_update=update,total_optimizer_updates=execution['optimizer_updates'],id=record['id'],view=view,
                rgb=float(rgb.detach()),mask=float(mask.detach()),elapsed_seconds=time.perf_counter()-started)
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(row,ensure_ascii=False),flush=True)
    save_checkpoint(model,optimizer,output/f"checkpoint_u{execution['optimizer_updates']:04d}.pt",
                    execution['optimizer_updates'],execution)
    return optimizer


def directional_check(model,batch,target,packet,raster,target_field,analytic):
    """同一标量目标沿法线输出bias梯度方向作一次中心差分。"""
    parameter = model.normal_head.cnn[-1].bias
    direction = parameter.grad.detach().clone()
    direction = direction/direction.norm().clamp_min(1e-12)
    original = parameter.detach().clone()
    values = []
    epsilon = 1e-3
    with torch.no_grad():
        for sign in (1.,-1.):
            parameter.copy_(original+sign*epsilon*direction)
            prediction = model(batch,snapshots=packet,raster=raster)
            values.append(float(geometry_losses(prediction,target,batch,target_field)[0]))
            del prediction
        parameter.copy_(original)
    expected = float((analytic*direction).sum())
    actual = (values[0]-values[1])/(2*epsilon)
    relative = abs(actual-expected)/max(abs(expected),abs(actual),1e-6)
    return dict(parameter='normal_head.cnn.2.bias',epsilon=epsilon,autograd=expected,
                finite_difference=actual,relative_error=relative,pass_check=relative<.1)


def main():
    """逐次真实update计账，profile包含在总预算中，不重置计数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--base-weights',required=True)
    parser.add_argument('--ft-checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--resolution',type=int,default=128)
    parser.add_argument('--sigma',type=float,default=2.)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--save-every',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--resume')
    parser.add_argument('--initialize-from')
    parser.add_argument('--geometry-optimizer-template')
    parser.add_argument('--experiment-id')
    parser.add_argument('--input-stage')
    parser.add_argument('--gradient-check',action='store_true')
    parser.add_argument('--appearance-updates',type=int,default=0)
    args = parser.parse_args()
    if args.initialize_from and args.resume:
        raise ValueError('共同参数初始化与同一几何阶段resume不能同时指定')
    if args.initialize_from and not (args.geometry_optimizer_template and args.experiment_id and args.input_stage):
        raise ValueError('新C分支需要真实几何优化器模板、experiment_id和input_stage')
    if not 1 <= args.updates or not 0 <= args.appearance_updates or args.updates+args.appearance_updates>2000:
        raise ValueError('首个候选总optimizer updates必须为1..2000，profile计入其中')
    from research.mentor_human_model.data.loader import load_manifest
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    setup_started = time.perf_counter()
    print(json.dumps(dict(stage='load_manifest_start',manifest=args.manifest)),flush=True)
    records = load_manifest(args.manifest,device='cpu',include_targets=True)
    print(json.dumps(dict(stage='model_construct_start',identities=len(records))),flush=True)
    model = MentorHumanModel(args.base_weights,args.ft_checkpoint,args.resolution,sigma=args.sigma)
    print(json.dumps(dict(stage='model_construct_complete')),flush=True)
    # 训练只复用固定主干的真实cache；完整infer从六图前向，不读cache。
    model.vggt.aggregator.cpu()
    for name,module in model.named_children():
        if name != 'vggt':
            module.cuda()
    model.vggt.depth_head.cuda()
    model.appearance.requires_grad_(False)
    optimizer_options = dict(lr=args.lr)
    if args.geometry_optimizer_template:
        template = torch.load(args.geometry_optimizer_template,map_location='cpu',weights_only=False,mmap=True)
        if template['actual_training']['appearance_updates'] != 0:
            raise ValueError('几何优化器模板不能来自appearance阶段')
        groups = template['optimizer']['param_groups']
        if len(groups) != 1:
            raise ValueError('本合同只支持真实已有的单组几何AdamW')
        # state_dict还保存AdamW固有的decoupled_weight_decay标记；它不是本机构造参数。
        options = ('lr','betas','eps','weight_decay','amsgrad','maximize','foreach','capturable','differentiable','fused')
        optimizer_options = {key:groups[0][key] for key in options}
        del template
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],**optimizer_options)
    start_update = 0
    inherited_updates = 0
    initialization_path = str(Path(args.initialize_from).resolve()) if args.initialize_from else None
    if args.resume or args.initialize_from:
        saved = torch.load(args.resume or args.initialize_from,map_location='cpu',weights_only=False)
        if saved['config'].get('sigma') != args.sigma:
            raise ValueError('checkpoint的DPSR sigma与该次运行不同，不能静默覆盖频率filter buffer续训')
        missing,unexpected = model.load_state_dict(saved['model'],strict=False)
        if unexpected or any(not k.startswith('vggt.aggregator.') for k in missing):
            raise RuntimeError(f'候选checkpoint缺失训练参数: {missing}, {unexpected}')
        if args.resume:
            if saved['actual_training'].get('appearance_updates',0):
                raise ValueError('appearance优化器不能作为几何resume状态')
            optimizer.load_state_dict(saved['optimizer'])
            start_update = saved['update']
            inherited_updates = saved['actual_training'].get('inherited_optimizer_updates',0)
            initialization_path = saved['actual_training'].get('initialize_from')
        else:
            inherited_updates = saved['update']
        del saved
    prepared = []
    cache_read_seconds = 0.
    feature_sources = []
    model.eval()
    for record in records:
        print(json.dumps(dict(stage='prepare_identity_start',id=record['id'])),flush=True)
        batch = move(record['batch'],'cuda')
        raster = model.prepare_raster(batch)
        print(json.dumps(dict(stage='prepare_raster_complete',id=record['id'])),flush=True)
        # 光栅只存固定源SMPL-X，不含scan；训练期间复用同一几何索引。
        packet = record['snapshots']
        if isinstance(packet,(str,Path)):
            feature_sources.append(dict(id=record['id'],path=str(packet),
                role='reused actual frozen RGB frame/global snapshots, original encoding cost belongs to producer'))
            cache_started = time.perf_counter()
            packet = torch.load(packet,map_location='cpu',weights_only=False,mmap=True)
            cache_read_seconds += time.perf_counter()-cache_started
        target = move(record['target'],'cuda')
        with torch.no_grad():
            target_field = model.surface.target(target['scan_points'],target['scan_normals'],
                batch['unit_center'].reshape(-1,3),batch['unit_scale'].reshape(-1)).detach()
        print(json.dumps(dict(stage='prepare_scan_target_complete',id=record['id'])),flush=True)
        prepared.append(dict(id=record['id'],batch=batch,target=target,raster=raster,
                             snapshots=packet,target_field=target_field))
    model.train()
    execution = dict(status='RUNNING',optimizer_updates=start_update,geometry_updates=start_update,appearance_updates=0,seed=args.seed,
        resume_checkpoint=str(Path(args.resume).resolve()) if args.resume else None,starting_update=start_update,
        manifest=str(args.manifest),fixed_backbone=str(args.base_weights),fixed_ft=str(args.ft_checkpoint),
        source_count=6,identity_count=len(records),ids=[r['id'] for r in records],resolution=args.resolution,sigma=args.sigma,
        prepared_seconds=time.perf_counter()-setup_started,cache_read_seconds=cache_read_seconds,
        frozen_backbone_execution='existing verified feature precompute reused; current optimizer loop does not run aggregator',
        frozen_feature_sources=feature_sources,
        denominator='depth/point/normal: actual valid source pixels; field: all R^3 voxels',
        weights=dict(depth=1,point=1,normal=.1,field=1),eikonal=False,
        frozen=['vggt.aggregator','appearance_until_geometry_endpoint'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.experiment_id:
        execution.update(new_arch=True,experiment_id=args.experiment_id,input_stage=args.input_stage,
            teacher_loss=False,pose_role='same_identity_same_pose_fixed_source_RGB_fit_SMPLX',
            train_subjects=[r['id'].split('/')[1] for r in records],
            camera_role='same_six_calibrated_source_K_E',eval_subjects=[],
            actual_updates=start_update,inherited_optimizer_updates=inherited_updates,
            initialize_from=initialization_path,
            geometry_optimizer_template=args.geometry_optimizer_template,geometry_optimizer_options=optimizer_options,
            fixed_incremental_endpoint=dict(geometry=1800,shared_appearance=200,total=2000),
            appearance_target='stage-consistent batch.images: OLD source RGB or PRT source RGB',
            comparison_scope='PRT lighting/render input adaptation including corresponding source RGB appearance supervision',
            draw_order='manifest order cyclic; appearance source view advances after one full identity cycle')
    json_write(out/'run.json',execution)
    started = time.perf_counter()
    baseline = {}
    try:
        if args.initialize_from:
            # U0只观察同一起点在当前输入上的目标，零更新节点不消耗训练预算。
            save_checkpoint(model,optimizer,out/'checkpoint_u0000.pt',0,execution)
            with torch.no_grad():
                zero_rows = []
                for record in prepared:
                    prediction = model(record['batch'],snapshots=record['snapshots'],raster=record['raster'])
                    total,losses = geometry_losses(prediction,record['target'],record['batch'],record['target_field'])
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError('共同B终点在当前输入的零更新目标非有限，保留U0停止')
                    zero_rows.append(dict(id=record['id'],actual_updates=0,total=float(total),
                        **{key:float(value) for key,value in losses.items()}))
                    json_write(out/'zero_update_losses.json',zero_rows)
                    del prediction,total,losses
        torch.cuda.reset_peak_memory_stats()
        with (out/'updates.jsonl').open('a',encoding='utf-8') as log:
            for update in range(start_update+1,args.updates+1):
                record = prepared[(update-1)%len(prepared)]
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prediction = model(record['batch'],snapshots=record['snapshots'],raster=record['raster'])
                total,losses = geometry_losses(prediction,record['target'],record['batch'],record['target_field'])
                if not bool(torch.isfinite(total)):
                    raise RuntimeError('真实总loss非有限，保存现场并停止该分支')
                total.backward()
                if update == 1:
                    grads = gradient_evidence(model)
                    json_write(out/'first_gradient.json',grads)
                    if any(not item['finite'] or item['norm']==0 for item in grads.values()):
                        raise RuntimeError('实际训练模块缺失或非有限梯度，禁止声称全链已训练')
                    if args.gradient_check:
                        check = directional_check(model,record['batch'],record['target'],record['snapshots'],
                            record['raster'],record['target_field'],model.normal_head.cnn[-1].bias.grad.detach().clone())
                        json_write(out/'directional_gradient.json',check)
                        if not check['pass_check']:
                            raise RuntimeError('同一scalar目标的参数方向梯度不符，停止修复')
                scalar = {k:float(v.detach()) for k,v in losses.items()}
                baseline.setdefault(record['id'],scalar)
                # 只响应本项目明确出现过的数量级失稳，不做自动调参/重复验证。
                if update>100 and (scalar['depth']>10*max(baseline[record['id']]['depth'],1e-4)
                                   or scalar['field']>10*max(baseline[record['id']]['field'],1e-4)):
                    raise RuntimeError('深度或共同field误差较本身份初值恶化一个数量级，保存停止')
                optimizer.step()
                torch.cuda.synchronize()
                execution['optimizer_updates'] = update
                execution['geometry_updates'] = update
                if args.experiment_id:
                    execution['actual_updates'] = execution['optimizer_updates']
                row = dict(update=update,id=record['id'],**scalar,total=float(total.detach()),
                    seconds=time.perf_counter()-t0,elapsed_seconds=time.perf_counter()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(row,ensure_ascii=False)+'\n')
                log.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)
                if update%args.save_every==0 or update==args.updates or update==20:
                    save_checkpoint(model,optimizer,out/f'checkpoint_u{update:04d}.pt',update,execution)
                    json_write(out/'run.json',execution)
                del prediction,total,losses
        if args.appearance_updates:
            optimizer = appearance_stage(model,prepared,args.appearance_updates,out,execution,started)
        execution['status'] = 'FIXED_ENDPOINT_COMPLETE_GEOMETRY_NOT_YET_ADJUDICATED'
        if args.experiment_id and args.updates < 1800:
            execution['status'] = 'PROFILE_COMPLETE_COUNTS_TOWARD_FIXED_ENDPOINT'
        if args.experiment_id:
            execution['actual_updates'] = execution['optimizer_updates']
        save_checkpoint(model,optimizer,out/f"checkpoint_u{execution['optimizer_updates']:04d}.pt",
                        execution['optimizer_updates'],execution)
    except Exception as exc:
        execution.update(status='STOPPED',failure=repr(exc))
        save_checkpoint(model,optimizer,out/'checkpoint_stopped.pt',execution['optimizer_updates'],execution)
        raise
    finally:
        execution['elapsed_seconds'] = time.perf_counter()-started
        json_write(out/'run.json',execution)


if __name__ == '__main__':
    main()
