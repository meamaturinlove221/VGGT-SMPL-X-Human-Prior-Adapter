"""固定B权重、标定与原始支持的完整输出评价。"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from research.dna_b_adapter_20261021.runtime import load_b_model
from research.dna_b_adapter_20261021.paired_train import load_arrays, write_json
from research.mentor_human_model.infer import save_source_raw, save_views, cube_statistics
from research.mentor_human_model.model.network import backproject
from research.mentor_human_model.model.surface import export_mesh
from research.mentor_human_model.model.gaussian import SurfaceGaussians


def measurement_metrics(prediction, target, batch):
    """只在同来源合法测量像素比较；缺测不置零，不使用预测置信度。"""
    valid=target['depth_valid'].bool()
    depth=(prediction['depth']-target['depth']).abs()[valid]
    world=backproject(target['depth'],batch['K'],batch['E'])
    direct=(prediction['world_points']-world).norm(dim=-1)[valid]
    fg=batch['masks'].bool()
    return dict(valid_pixels=int(valid.sum()),foreground_pixels=int(fg.sum()),
                missing_measurement_foreground=int((fg & ~valid).sum()),
                depth_MAE=float(depth.mean()),depth_P95=float(torch.quantile(depth,.95)),
                Direct_mean=float(direct.mean()),Direct_P95=float(torch.quantile(direct,.95)),
                units='source calibrated world units; no fitted alignment',
                raw_foreground_count=int(fg.sum()))


def count_dpsr(module, inputs):
    """仅计实际进入的原DPSR调用，不将cube前置拒绝算为求解。"""
    module.actual_followup_calls += 1


@torch.no_grad()
def evaluate(checkpoint, manifest_path, output):
    """每个实际输入只执行一次B，场失败仍保留全部RAW及失败阶段。"""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if 'samples' in manifest:
        samples=manifest['samples']
    else:
        samples=manifest['fixed16']+manifest['hard']+manifest['source_fitted']+manifest['th_control']
    torch.set_num_threads(4)
    model,saved=load_b_model(checkpoint)
    del saved
    for name,module in model.named_children():
        if name!='vggt': module.cuda()
    model.vggt.depth_head.cuda();model.eval()
    model.surface.dpsr.actual_followup_calls=0
    model.surface.dpsr.register_forward_pre_hook(count_dpsr)
    records=[]
    for item in samples:
        started=time.perf_counter()
        directory=output/item['id'].replace('/','__')/item.get('prior_role','GIVEN_PRIOR')
        directory.mkdir(parents=True,exist_ok=True)
        batch=load_arrays(item['input_npz'],'cuda');batch['smpl_faces']=batch['smpl_faces'].long()
        packet=torch.load(item['snapshots'],map_location='cpu',weights_only=False,mmap=True)
        write_json(directory/'FORWARD_CALL.json',dict(started=True,completed=False,optimizer_updates=0))
        prediction=model.forward_source_depth(batch,snapshots=packet)
        write_json(directory/'FORWARD_CALL.json',dict(started=True,completed=True,optimizer_updates=0))
        valid=prediction['normal_valid'][:,:,::4,::4]
        points=prediction['world_points'][:,:,::4,::4][valid].reshape(1,-1,3)
        normals=prediction['world_normal'][:,:,::4,::4][valid].reshape(1,-1,3)
        prediction.update(oriented_points=points,oriented_normals=normals)
        row=dict(id=item['id'],prior_role=item.get('prior_role','GIVEN_PRIOR'),
                 checkpoint=str(Path(checkpoint).resolve()),input_npz=item['input_npz'],
                 camera_role='REFERENCE_INPUT_CALIBRATION',network_forwards=1,
                 aggregator_forwards=0,optimizer_updates=0,output=str(directory),
                 frozen_Xf=item['snapshots'],cube=cube_statistics(points,batch['unit_center'],batch['unit_scale']))
        target=load_arrays(item['target_npz'],'cuda')
        row['metrics']=measurement_metrics(prediction,target,batch)
        dpsr_before=model.surface.dpsr.actual_followup_calls
        try:
            geometry=model.surface(points,normals,batch['unit_center'],batch['unit_scale'])
            prediction.update(geometry)
            vertices,faces=export_mesh(geometry['indicator'],batch['unit_center'].cpu().numpy(),
                                       batch['unit_scale'].cpu().numpy(),directory/'clothed_mesh.ply')
            export_mesh(geometry['indicator_base'],batch['unit_center'].cpu().numpy(),
                        batch['unit_scale'].cpu().numpy(),directory/'base_mesh.ply')
            np.savez_compressed(directory/'complete_mesh.npz',vertices=vertices,faces=faces)
            gs=SurfaceGaussians.from_mesh(vertices,faces,max_gaussians=12000).cuda().requires_grad_(False)
            gs.initialize_colors(batch['images'][0].permute(0,2,3,1),batch['masks'][0],
                                 batch['E'][0],batch['K'][0],depths=prediction['depth'][0])
            appearance=model.appearance(gs,batch['unit_center'],batch['unit_scale'])
            gs.save_ply(directory/'clothed_gaussians.ply',appearance=appearance)
            height,width=batch['images'].shape[-2:]
            for view in range(6):
                rendered=gs.render(batch['E'][0,view:view+1],batch['K'][0,view:view+1],width,height,appearance=appearance)
                arrays={key:value[0].cpu().numpy() for key,value in rendered.items()}
                save_views(directory/f'gs_source_{view:02d}',arrays['rgb'],arrays['depth'],arrays['normal'],arrays['mask'])
            row['surface_status']='EXPORTED_ALL_COMPONENTS';row['gaussian_renders']=6
            del gs,appearance,rendered
        except ValueError as error:
            row['surface_status']='FAILED_RAW_PRESERVED';row['surface_error']=str(error)
        row['surface_decoder_attempts']=1
        row['DPSR_core_forwards']=model.surface.dpsr.actual_followup_calls-dpsr_before
        source={key:value.detach().float().cpu().numpy() for key,value in prediction.items() if torch.is_tensor(value)}
        save_source_raw(directory,source,batch)
        row['seconds']=time.perf_counter()-started
        write_json(directory/'INFERENCE.json',row)
        records.append(row);write_json(output/'EVALUATION_RECEIPT.json',records)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        del prediction,packet,batch,target,source,points,normals
        torch.cuda.empty_cache()
    return records
