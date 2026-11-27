"""固定 FT 双流特征、全顶点稀疏先验与共同穿衣几何的正式候选。"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
EXISTING_OPS = Path(__file__).resolve().parent / 'existing_ops'


def existing_module(name, relative):
    """直接复用已存在的真实算子，不复制第三方或换成稠密伪实现。"""
    spec = importlib.util.spec_from_file_location(name, EXISTING_OPS / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SPARSE = existing_module('mentor_candidate_existing_sparse', 'sparse_encoder.py')
P3D = existing_module('mentor_candidate_existing_p3d', 'p3d_bridge.py')


def vertex_normals(vertices, faces):
    """全顶点面积加权法线；不删除衣服或难点。"""
    result = torch.zeros_like(vertices)
    if faces.ndim == 3:
        faces = faces[0]
    for b in range(len(vertices)):
        a, c, d = (vertices[b, faces[:, i]] for i in range(3))
        normal = torch.cross(c - a, d - a, dim=-1)
        for i in range(3):
            result[b].index_add_(0, faces[:, i], normal)
    return F.normalize(result, dim=-1, eps=1e-6)


def backproject(depth, K, E):
    """camera-z 与 world→camera 在同一尺度中反投影，像素坐标沿用现有协议。"""
    h, w = depth.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                            torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    x = (xx - K[..., 0, 2, None, None]) * depth / K[..., 0, 0, None, None]
    y = (yy - K[..., 1, 2, None, None]) * depth / K[..., 1, 1, None, None]
    camera = torch.stack((x, y, depth), -1)
    return torch.einsum('bshwj,bsjk->bshwk', camera - E[..., :3, 3][:, :, None, None], E[..., :3, :3])


def depth_normals(points, masks):
    """由实际穿衣深度构造朝向可见相机的世界法线。"""
    dx = points[:, :, 1:-1, 2:] - points[:, :, 1:-1, :-2]
    dy = points[:, :, 2:, 1:-1] - points[:, :, :-2, 1:-1]
    normal = F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1e-6)
    valid = (masks[:, :, 1:-1, 1:-1] & masks[:, :, 1:-1, 2:] & masks[:, :, 1:-1, :-2]
             & masks[:, :, 2:, 1:-1] & masks[:, :, :-2, 1:-1])
    b, s, h, w, _ = points.shape
    padded = F.pad(normal.reshape(b*s,h-2,w-2,3).permute(0,3,1,2), (1,1,1,1))
    return padded.permute(0,2,3,1).reshape(b,s,h,w,3), F.pad(valid, (1,1,1,1))


def source_camera_scale(E):
    """复用既有reference-camera协议的六标定相机中心RMS半径，不读GT。"""
    centers = -(E[...,:3,:3].transpose(-1,-2) @ E[...,:3,3,None]).squeeze(-1)
    return (centers-centers.mean(1,keepdim=True)).square().sum(-1).mean(1).sqrt()


class LearnedFusion(nn.Module):
    """真实 frame/global Xf 与 P3D concat 后经 MLP，保留多层独立出口。"""
    def __init__(self):
        super().__init__()
        self.mixers = nn.ModuleDict({str(i): nn.Sequential(nn.LayerNorm(3072), nn.Linear(3072,256),
            nn.GELU(), nn.Linear(256,2048)) for i in (4,11,17,23)})
        for mixer in self.mixers.values():
            nn.init.normal_(mixer[-1].weight, std=1e-4)
            nn.init.zeros_(mixer[-1].bias)

    def forward(self, index, dual, prior):
        """小非零初始化使首个梯度检查即可到达稀疏先验，不声称与 FT 逐值相同。"""
        xf1, xf2 = dual.chunk(2, dim=-1)
        return dual + self.mixers[str(index)](torch.cat((xf1, xf2, prior), dim=-1)).to(dual.dtype)


class ClothingNormalHead(nn.Module):
    """几何法线加零初始化残差，学习衣服而非把裸体法线作为目标。"""
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(2048,16)
        self.cnn = nn.Sequential(nn.Conv2d(23,32,3,padding=1),nn.GELU(),nn.Conv2d(32,3,1))
        nn.init.zeros_(self.cnn[-1].weight)
        nn.init.zeros_(self.cnn[-1].bias)

    def forward(self, dual, images, depth, normal, patch_start):
        """共享真实融合特征、RGB、深度与几何法线生成世界法线残差。"""
        b,s,_,h,w = images.shape
        grid = self.feature(dual[:,:,patch_start:].float()).reshape(b*s,h//14,w//14,16).permute(0,3,1,2)
        grid = F.interpolate(grid, size=(h,w),mode='bilinear',align_corners=False)
        n = normal.reshape(b*s,h,w,3).permute(0,3,1,2)
        source = torch.cat((images.reshape(b*s,3,h,w), depth.reshape(b*s,1,h,w),n,grid),1)
        residual = self.cnn(source).permute(0,2,3,1).reshape(b,s,h,w,3)
        return F.normalize(normal + residual, dim=-1, eps=1e-6), residual


class MentorHumanModel(nn.Module):
    """推理只消费六图、源mask、相机和SMPL-X；scan只允许进入独立loss。"""
    def __init__(self, base_weights, ft_checkpoint, resolution=128, vertex_count=10475, sigma=2.0):
        super().__init__()
        from vggt.models.vggt import VGGT
        from .surface import GeometryDecoder
        from .gaussian import SurfaceAppearance
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_track=False,
            enable_human_prior_fusion=False, enable_human_prior_summary=False)
        base = torch.load(base_weights,map_location='cpu',weights_only=True,mmap=True)
        self.vggt.aggregator.load_state_dict({k.removeprefix('aggregator.'):v for k,v in base.items()
            if k.startswith('aggregator.')},strict=True)
        del base
        ft = torch.load(ft_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        self.vggt.depth_head.load_state_dict(ft['depth_head'],strict=True)
        del ft
        self.vggt.aggregator.eval().requires_grad_(False)
        self.vertex_features = nn.Embedding(vertex_count,16)
        nn.init.normal_(self.vertex_features.weight,std=0.01)
        self.sparse = SPARSE.SMPLXSparseConvFeatureEncoder(in_dim=22,hidden_dim=32,out_dim=32,
            num_layers=2,grid_size=64,bounds=(-1.,1.),backend='torch',block_type='submanifold_cross')
        self.prior_bridge = nn.Linear(32,1024)
        self.fusion = LearnedFusion()
        self.normal_head = ClothingNormalHead()
        self.surface = GeometryDecoder(resolution=resolution,sigma=sigma)
        self.appearance = SurfaceAppearance()
        self.config = dict(base_weights=str(base_weights),ft_checkpoint=str(ft_checkpoint),resolution=resolution,sigma=float(sigma),
                           vertex_count=vertex_count,prior_backend='torch_submanifold_7tap')

    def train(self, mode=True):
        """固定图像骨干始终eval，其余正式几何模块可训练。"""
        super().train(mode)
        self.vggt.aggregator.eval()
        return self

    @torch.no_grad()
    def encode_images(self, images):
        """一次真实冻结 VGGT 执行，缓存仅为四层 frame/global 特征。"""
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
            snapshots, start = self.vggt.aggregator(images)
        return dict(snapshots={i:snapshots[i].detach().cpu() for i in (4,11,17,23)},patch_start=int(start))

    @torch.no_grad()
    def prepare_raster(self, batch):
        """现有三角形zbuffer固定几何索引；之后特征插值保留梯度。"""
        h,w = batch['images'].shape[-2:]
        E = batch['E']
        return P3D.prepare_dense_mesh_raster_cache_vectorized(batch['smpl_vertices'],batch['smpl_faces'],
            batch['K'],E[...,:3,:3],E[...,:3,3],image_height=h,image_width=w,patch_size=14)

    def forward_source_depth(self, batch, snapshots=None, raster=None, prior_mode='normal'):
        """完整六源先验/Fusion/DPT原深度路径；DNA唯depth目标不求未监督DPSR场。"""
        images,masks = batch['images'], batch['masks'].bool()
        if images.shape[0] != 1:
            raise ValueError('首个完整候选固定batch=1，避免混合不同人体DPSR点集')
        if images.shape[1] != 6:
            raise ValueError('正式候选要求六源视角；四源属于单独消融')
        if snapshots is None:
            snapshots = self.encode_images(images)
        start = snapshots['patch_start']
        native = [None]*24
        for i,tensor in snapshots['snapshots'].items():
            native[int(i)] = tensor.to(images.device)
        center,scale = batch['unit_center'].reshape(-1,3),batch['unit_scale'].reshape(-1)
        vertices = batch['smpl_vertices']
        xyz = (vertices-center[:,None]) / scale[:,None,None] * 2
        normals = vertex_normals(vertices,batch['smpl_faces'])
        feature = self.vertex_features.weight[None].expand(len(vertices),-1,-1)
        encoded = self.sparse(xyz,torch.cat((xyz,normals,feature),-1))['encoded_point_features']
        if raster is None:
            raster = self.prepare_raster(batch)
        prior = P3D.apply_dense_mesh_raster_cache(encoded,raster,token_bridge=self.prior_bridge)['mentor_dual_xf_prior_tokens']
        if prior_mode == 'zero':
            prior = prior*0
        elif prior_mode == 'shuffle':
            prior = prior.flip(2)
        elif prior_mode != 'normal':
            raise ValueError('prior_mode must be normal/zero/shuffle')
        fused = [None]*24
        for i in (4,11,17,23):
            with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
                patches = self.fusion(i,native[i][:,:,start:],prior.to(native[i].dtype))
            fused[i] = torch.cat((native[i][:,:,:start],patches),2)
        with torch.autocast(images.device.type,enabled=False):
            if self.training and torch.is_grad_enabled():
                depth,confidence = checkpoint(self.vggt.depth_head,fused,images,start,1,use_reentrant=False)
            else:
                depth,confidence = self.vggt.depth_head(fused,images,start,frames_chunk_size=1)
            normalized_depth = depth[...,0]
            calibrated_scale = source_camera_scale(batch['E'])
            depth = normalized_depth*calibrated_scale[:,None,None,None]
            world = backproject(depth,batch['K'],batch['E'])
            geometric,valid = depth_normals(world,masks)
            normal,residual = self.normal_head(fused[23],images,depth,geometric,start)
            # 固定均匀像素子采样仅供完整DPSR；源深度与RAW仍保留全部像素。
            support = valid[:, :, ::4, ::4]
            p = world[:,:,::4,::4][support].reshape(1,-1,3)
            n = normal[:,:,::4,::4][support].reshape(1,-1,3)
        return dict(depth=depth,normalized_depth=normalized_depth,source_camera_scale=calibrated_scale,
                    depth_conf=confidence,world_points=world,world_normal=normal,
                    geometric_normal=geometric,normal_residual=residual,normal_valid=valid,
                    oriented_points=p,oriented_normals=n)

    def forward(self, batch, snapshots=None, raster=None, prior_mode='normal'):
        """正常/零/打乱先验保持全部其它路径相同，完整源RAW永不删除。"""
        images,masks = batch['images'], batch['masks'].bool()
        if images.shape[0] != 1:
            raise ValueError('首个完整候选固定batch=1，避免混合不同人体DPSR点集')
        if images.shape[1] != 6:
            raise ValueError('正式候选要求六源视角；四源属于单独消融')
        if snapshots is None:
            snapshots = self.encode_images(images)
        start = snapshots['patch_start']
        native = [None]*24
        for i,tensor in snapshots['snapshots'].items():
            native[int(i)] = tensor.to(images.device)
        center,scale = batch['unit_center'].reshape(-1,3),batch['unit_scale'].reshape(-1)
        vertices = batch['smpl_vertices']
        xyz = (vertices-center[:,None]) / scale[:,None,None] * 2
        normals = vertex_normals(vertices,batch['smpl_faces'])
        feature = self.vertex_features.weight[None].expand(len(vertices),-1,-1)
        encoded = self.sparse(xyz,torch.cat((xyz,normals,feature),-1))['encoded_point_features']
        if raster is None:
            raster = self.prepare_raster(batch)
        prior = P3D.apply_dense_mesh_raster_cache(encoded,raster,token_bridge=self.prior_bridge)['mentor_dual_xf_prior_tokens']
        if prior_mode == 'zero':
            prior = prior*0
        elif prior_mode == 'shuffle':
            prior = prior.flip(2)
        elif prior_mode != 'normal':
            raise ValueError('prior_mode must be normal/zero/shuffle')
        fused = [None]*24
        for i in (4,11,17,23):
            with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
                patches = self.fusion(i,native[i][:,:,start:],prior.to(native[i].dtype))
            fused[i] = torch.cat((native[i][:,:,:start],patches),2)
        with torch.autocast(images.device.type,enabled=False):
            if self.training and torch.is_grad_enabled():
                depth,confidence = checkpoint(self.vggt.depth_head,fused,images,start,1,use_reentrant=False)
            else:
                depth,confidence = self.vggt.depth_head(fused,images,start,frames_chunk_size=1)
            normalized_depth = depth[...,0]
            calibrated_scale = source_camera_scale(batch['E'])
            depth = normalized_depth*calibrated_scale[:,None,None,None]
            world = backproject(depth,batch['K'],batch['E'])
            geometric,valid = depth_normals(world,masks)
            normal,residual = self.normal_head(fused[23],images,depth,geometric,start)
            # 固定均匀像素子采样仅供DPSR；完整RAW与源监督保留全部像素。
            support = valid[:, :, ::4, ::4]
            p = world[:,:,::4,::4][support].reshape(1,-1,3)
            n = normal[:,:,::4,::4][support].reshape(1,-1,3)
            geometry = self.surface(p,n,center,scale)
        return dict(depth=depth,normalized_depth=normalized_depth,source_camera_scale=calibrated_scale,
                    depth_conf=confidence,world_points=world,world_normal=normal,
                    geometric_normal=geometric,normal_residual=residual,normal_valid=valid,
                    oriented_points=p,oriented_normals=n,**geometry)
