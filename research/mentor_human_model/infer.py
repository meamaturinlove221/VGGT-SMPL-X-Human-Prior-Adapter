"""六源RGB真实前向，输出完整RAW/mesh/真实Gaussian/纯GS查询与360。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.mentor_human_model.model.network import MentorHumanModel
from research.mentor_human_model.model.surface import export_mesh
from research.mentor_human_model.model.gaussian import SurfaceGaussians
from research.mentor_human_model.model.network import backproject, depth_normals, source_camera_scale


def point_ply(path,points,colors):
    """写所有源前景点，保留跨视图重影和离群点。"""
    points = np.asarray(points,dtype='<f4')
    colors = np.clip(np.asarray(colors)*255,0,255).astype(np.uint8)
    array = np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,name in enumerate(('x','y','z')):
        array[name] = points[:,i]
    for i,name in enumerate(('red','green','blue')):
        array[name] = colors[:,i]
    with open(path,'wb') as file:
        file.write(('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(array)}\n'+
            'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii'))
        file.write(array.tobytes())


def image_write(path,array):
    """显示用PNG不替代原始浮点数组。"""
    Image.fromarray(np.clip(np.asarray(array)*255,0,255).astype(np.uint8)).save(path)


def save_views(directory,rgb,depth,normal,mask):
    """每帧保留真实RGB/depth/world normal/alpha，分别写NPZ和显示PNG。"""
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'render.npz',rgb=rgb,depth=depth,world_normal=normal,mask=mask)
    image_write(directory/'rgb.png',rgb)
    image_write(directory/'normal.png',(normal+1)*.5*mask[...,None])
    image_write(directory/'mask.png',mask)
    supported = depth[mask>.01]
    if len(supported):
        lo,hi = np.percentile(supported,[1,99])
        image_write(directory/'depth.png',(depth-lo)/max(hi-lo,1e-6)*(mask>.01))


@torch.no_grad()
def export_ft_baseline(model,batch,packet,directory,checkpoint):
    """同次真实六图特征供固定FT对照，完整保留RAW并恢复候选DPT。"""
    candidate = {key:value.detach().cpu().clone() for key,value in model.vggt.depth_head.state_dict().items()}
    fixed = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    try:
        model.vggt.depth_head.load_state_dict(fixed['depth_head'],strict=True)
        native = [None]*24
        for index,tensor in packet['snapshots'].items():
            native[int(index)] = tensor.to(batch['images'].device)
        depth,confidence = model.vggt.depth_head(native,batch['images'],packet['patch_start'],frames_chunk_size=1)
        normalized = depth[...,0]
        scale = source_camera_scale(batch['E'])
        depth = normalized*scale[:,None,None,None]
        world = backproject(depth,batch['K'],batch['E'])
        normal,valid = depth_normals(world,batch['masks'].bool())
        tensors = dict(depth=depth,normalized_depth=normalized,source_camera_scale=scale,
            depth_conf=confidence,world_points=world,world_normal=normal,normal_valid=valid,
            K=batch['K'],E=batch['E'],mask=batch['masks'],unit_center=batch['unit_center'],unit_scale=batch['unit_scale'])
        arrays = {key:value.detach().float().cpu().numpy() for key,value in tensors.items()}
        directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'source_RAW.npz',**arrays)
        rgb = batch['images'][0].permute(0,2,3,1).cpu().numpy()
        masks = batch['masks'][0].bool().cpu().numpy()
        point_ply(directory/'source_RAW_all_foreground.ply',arrays['world_points'][0][masks],rgb[masks])
        (directory/'BASELINE.json').write_text(json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()),
            method='VGGT_FT_U800 depth head + fixed official backbone',
            inputs='same actual six-RGB aggregator packet as candidate; no identity feature cache',
            scale='same source camera RMS camera-z world recovery',optimizer_updates=0,
            training_overlap='historical FT may have seen DNA; not a never-seen claim'),indent=2),encoding='utf-8')
    finally:
        model.vggt.depth_head.load_state_dict(candidate,strict=True)


def save_source_raw(out,source,batch):
    """几何场失败也保留未裁剪的全部源输出。"""
    source.update(K=batch['K'].cpu().numpy(),E=batch['E'].cpu().numpy(),mask=batch['masks'].cpu().numpy(),
                  unit_center=batch['unit_center'].cpu().numpy(),unit_scale=batch['unit_scale'].cpu().numpy())
    np.savez_compressed(out/'source_RAW.npz',**source)
    rgb = batch['images'][0].permute(0,2,3,1).cpu().numpy()
    masks = batch['masks'][0].bool().cpu().numpy()
    point_ply(out/'source_RAW_all_foreground.ply',source['world_points'][0][masks],rgb[masks])
    for s in range(6):
        save_views(out/f'source_{s:02d}',rgb[s],source['depth'][0,s],source['world_normal'][0,s],masks[s].astype(float))


def cube_statistics(points,center,scale):
    """记录输入cube内外真实范围，不裁剪坐标。"""
    unit = (points-center.reshape(1,1,3))/scale.reshape(1,1,1)+.5
    return dict(point_count=unit.shape[1],unit_min=unit.amin((0,1)).cpu().tolist(),
        unit_max=unit.amax((0,1)).cpu().tolist(),outside_count=int(((unit<=0)|(unit>=1)).any(-1).sum().item()),
        rule='strict interior (0,1); original coordinates retained; no clipping/cube change')


def main():
    """查询只读取相机，推理从未读取scan、targets或训练RGB特征cache。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--query-cameras',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--lineage',help='可选MANIFEST.json，仅取该输入对应的文字来源说明')
    parser.add_argument('--prior-mode',choices=['normal','zero','shuffle'],default='normal')
    parser.add_argument('--export-ft-baseline',action='store_true',help='同次真实六图编码导出固定FT RAW对照')
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    model = MentorHumanModel(**{k:v for k,v in saved['config'].items() if k!='prior_backend'})
    missing,unexpected = model.load_state_dict(saved['model'],strict=False)
    if unexpected or any(not key.startswith('vggt.aggregator.') for key in missing):
        raise RuntimeError(f'checkpoint缺失候选训练模块: {missing}, {unexpected}')
    model.cuda().eval()
    with np.load(args.inputs,allow_pickle=False) as archive:
        batch = {key:torch.as_tensor(archive[key],device='cuda') for key in archive.files}
    # 真实历史DNA存档的face为int32，P3D索引接口固定int64；拓扑与数值不变。
    batch['smpl_faces'] = batch['smpl_faces'].long()
    # 推理包含真实六图aggregator，仅训练可复用已核验固定特征。
    with torch.no_grad():
        torch.cuda.synchronize()
        encode_start = time.perf_counter()
        packet = model.encode_images(batch['images'])
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter()-encode_start
        print(json.dumps(dict(stage='fresh_six_rgb_encoded',seconds=encode_seconds)),flush=True)
        model.vggt.aggregator.cpu()
        torch.cuda.empty_cache()
        if args.export_ft_baseline:
            export_ft_baseline(model,batch,packet,out/'baseline_FT',saved['config']['ft_checkpoint'])
            print(json.dumps(dict(stage='fixed_ft_raw_exported')),flush=True)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        captured = {}
        def capture_depth(module,inputs,output):
            """留住共同场之前的真实DPT输出，供失败时完整导出。"""
            captured['depth'] = output
        def capture_normal(module,inputs,output):
            """留住已恢复normal头的原始预测，不生成替代法线。"""
            captured['normal'] = output
        def capture_cube(module,inputs):
            """进入原解码器前只记录真实点范围，不修改点或cube。"""
            captured['cube'] = cube_statistics(inputs[0],inputs[2],inputs[3])
            (out/'CUBE_DIAGNOSTIC.json').write_text(json.dumps(captured['cube'],indent=2),encoding='utf-8')
        handles = [model.vggt.depth_head.register_forward_hook(capture_depth),
            model.normal_head.register_forward_hook(capture_normal),model.surface.register_forward_pre_hook(capture_cube)]
        try:
            prediction = model(batch,snapshots=packet,prior_mode=args.prior_mode)
        except ValueError as error:
            if 'cube' not in captured or 'depth' not in captured or 'normal' not in captured:
                raise
            normalized = captured['depth'][0][...,0]
            calibrated_scale = source_camera_scale(batch['E'])
            depth = normalized*calibrated_scale[:,None,None,None]
            world = backproject(depth,batch['K'],batch['E'])
            normal,valid = depth_normals(world,batch['masks'].bool())
            tensors = dict(depth=depth,normalized_depth=normalized,source_camera_scale=calibrated_scale,
                depth_conf=captured['depth'][1],world_points=world,world_normal=captured['normal'][0],
                normal_residual=captured['normal'][1],geometric_normal=normal,normal_valid=valid)
            save_source_raw(out,{key:value.detach().float().cpu().numpy() for key,value in tensors.items()},batch)
            torch.cuda.synchronize()
            failure = dict(status='COMMON_FIELD_FAILED_RAW_RETAINED',error=str(error),checkpoint=str(Path(args.checkpoint).resolve()),
                input_npz=str(Path(args.inputs).resolve()),optimizer_updates_this_run=0,cube=captured['cube'],
                fresh_six_rgb_encode_seconds=encode_seconds,wall_seconds=time.perf_counter()-started,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            (out/'INFERENCE_FAILED.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
            raise
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter()-forward_start
        print(json.dumps(dict(stage='candidate_forward',seconds=forward_seconds)),flush=True)
    source = {key:value.detach().float().cpu().numpy() for key,value in prediction.items() if torch.is_tensor(value)}
    save_source_raw(out,source,batch)
    rgb = batch['images'][0].permute(0,2,3,1).cpu().numpy()
    masks = batch['masks'][0].bool().cpu().numpy()
    point_ply(out/'source_SMPLX_vertices.ply',batch['smpl_vertices'][0].cpu().numpy(),
              np.full((batch['smpl_vertices'].shape[1],3),.5))
    vertices,faces = export_mesh(prediction['indicator'],source['unit_center'],source['unit_scale'],out/'clothed_mesh.ply')
    np.savez_compressed(out/'complete_mesh.npz',vertices=vertices,faces=faces,indicator=source['indicator'],
                        indicator_base=source['indicator_base'],unit_center=source['unit_center'],unit_scale=source['unit_scale'])
    gs = SurfaceGaussians.from_mesh(vertices,faces,max_gaussians=12000).cuda().requires_grad_(False)
    with torch.no_grad():
        gs.initialize_colors(batch['images'][0].permute(0,2,3,1),batch['masks'][0],
                             batch['E'][0],batch['K'][0],depths=prediction['depth'][0])
        appearance = model.appearance(gs,batch['unit_center'],batch['unit_scale'])
        gs.save_ply(out/'clothed_gaussians.ply',appearance=appearance)
    np.savez_compressed(out/'complete_gaussians.npz',centers=gs.centers.cpu().numpy(),normals=gs.normals.cpu().numpy(),
        quaternion_wxyz=gs.quaternions.cpu().numpy(),**{key:value.detach().cpu().numpy() for key,value in appearance.items()})
    cameras = json.loads(Path(args.query_cameras).read_text(encoding='utf-8'))['cameras']
    if args.export_ft_baseline:
        from research.mentor_human_model.evaluation.dna_transfer import export_mesh_first_surface
        export_mesh_first_surface(vertices,faces,cameras[:2],out,device='cuda')
        print(json.dumps(dict(stage='complete_mesh_first_surface_exported',query_count=min(2,len(cameras)))),flush=True)
    h,w = batch['images'].shape[-2:]
    frames = []
    camera_rows = []
    with torch.no_grad():
        for i,camera in enumerate(cameras):
            K = torch.as_tensor(camera['K'],device='cuda',dtype=torch.float32)[None]
            E = torch.as_tensor(camera['E'],device='cuda',dtype=torch.float32)[None]
            query_w,query_h = int(camera.get('width',w)),int(camera.get('height',h))
            rendered = gs.render(E,K,query_w,query_h,appearance=appearance)
            arrays = {key:value[0].detach().cpu().numpy() for key,value in rendered.items()}
            save_views(out/f'query_{i:03d}',arrays['rgb'],arrays['depth'],arrays['normal'],arrays['mask'])
            frames.append(np.clip(arrays['rgb']*255,0,255).astype(np.uint8))
            equal_source = any(torch.allclose(E[0],old,atol=1e-5,rtol=1e-5) for old in batch['E'][0])
            camera_rows.append(dict(camera,index=i,coincides_with_source_camera=equal_source,
                                    render_role='source_camera_in_360' if equal_source else 'non_input_camera_query'))
            print(json.dumps(dict(stage='query_render',index=i,total=len(cameras))),flush=True)
    import imageio.v2 as imageio
    # 真实留出相机可以有独立尺寸；视频统一显示尺寸不改变保存的原始render数组。
    video_frames = [np.asarray(Image.fromarray(frame).resize((w,h))) if frame.shape[:2]!=(h,w) else frame for frame in frames]
    imageio.mimsave(out/'gaussian_360.mp4',video_frames,fps=6,macro_block_size=1)
    Image.fromarray(np.concatenate(video_frames[::max(1,len(video_frames)//6)][:6],axis=1)).save(out/'gaussian_360_contact.png')
    lineage = {}
    if args.lineage:
        manifest = json.loads(Path(args.lineage).read_text(encoding='utf-8'))
        for sample in manifest['samples']:
            candidate = (Path(args.lineage).parent/sample['input_npz']).resolve()
            if candidate == Path(args.inputs).resolve():
                lineage = sample['lineage']
    evidence = dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_update=saved['update'],
        training=saved['actual_training'],input_npz=str(Path(args.inputs).resolve()),source_views=6,
        query_input='camera_K_E_only; no query RGB/mask/depth read',source_lineage=lineage,
        camera_scale='source-provided calibrated world, isotropic source SMPL-X cube',
        raw='all source pixels NPZ; all source-mask points PLY; no filtering/component deletion',
        mesh='ordinary marching cubes export only, all components retained',gaussian_count=len(gs.centers),
        appearance='shared source-conditioned MLP, no inference optimization',
        gaussian_renderer='native Torch anisotropic EWA, depth sorted front-to-back alpha',
        geometry_passed=False,BODY_LOCKED=False,prior_mode=args.prior_mode,cameras=camera_rows)
    torch.cuda.synchronize()
    evidence.update(runtime=dict(wall_seconds=time.perf_counter()-started,fresh_six_rgb_encode_seconds=encode_seconds,
        candidate_forward_seconds=forward_seconds,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name(0)),
        optimizer_updates_this_run=0,ft_baseline_exported=args.export_ft_baseline,
        checkpoint_restore='all candidate trained parameters restored; frozen aggregator restored from fixed base checkpoint',
        gaussian_anchors='deterministic complete predicted mesh area sampling, seed4747001; shared learned appearance restored')
    (out/'INFERENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
