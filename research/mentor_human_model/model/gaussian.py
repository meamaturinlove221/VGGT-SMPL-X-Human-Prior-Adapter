"""True anisotropic 3D Gaussians anchored to frozen learned clothed mesh.

Rasterization: gsplat, world-to-camera OpenCV convention. No image decoder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SurfaceGaussians(nn.Module):
    def __init__(self, centers, normals, quaternions, scales, colors):
        super().__init__()
        self.register_buffer('centers', torch.as_tensor(centers, dtype=torch.float32))
        self.register_buffer('normals', torch.as_tensor(normals, dtype=torch.float32))
        self.register_buffer('quaternions', torch.as_tensor(quaternions, dtype=torch.float32))
        self.log_scales = nn.Parameter(torch.as_tensor(scales, dtype=torch.float32).log())
        self.opacity_logits = nn.Parameter(torch.full((len(centers),), 2.1972246))
        rgb = torch.as_tensor(colors, dtype=torch.float32).clamp(0.001, 0.999)
        self.color_logits = nn.Parameter(torch.logit(rgb))

    @classmethod
    def from_mesh(cls, vertices, faces, colors=None, max_gaussians=30000, seed=4747001):
        """Deterministic area-stratified anchors; no component/confidence filtering.

        The source mesh remains complete. This parameter is anchor sampling
        density, never a post-hoc deletion of learned Gaussians.
        """
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int64)
        triangles = v[f]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = np.linalg.norm(cross, axis=1) * 0.5
        if not area.sum() > 0:
            raise ValueError('Mesh has no nondegenerate area for Gaussian anchors.')
        count = int(max_gaussians)
        selected = np.searchsorted(np.cumsum(area), (np.arange(count) + 0.5) / count * area.sum())
        rng = np.random.default_rng(seed)
        uv = rng.random((count, 2))
        a = np.sqrt(uv[:, 0])
        barycentric = np.stack((1-a, a*(1-uv[:, 1]), a*uv[:, 1]), axis=-1)
        centers = (triangles[selected] * barycentric[..., None]).sum(1)
        normals = cross[selected] / (2 * area[selected, None])
        tangent = triangles[selected, 1] - triangles[selected, 0]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        bitangent = np.cross(normals, tangent)
        rotation = np.stack((tangent, bitangent, normals), axis=-1)
        quaternions = matrix_quaternion(torch.from_numpy(rotation)).numpy()
        radius = np.sqrt(area.sum() / count / np.pi)
        scales = np.tile([radius, radius, radius * 0.1], (count, 1)).astype(np.float32)
        if colors is None:
            rgb = np.full((count, 3), 0.5, dtype=np.float32)
        else:
            rgb = (np.asarray(colors)[f[selected]] * barycentric[..., None]).sum(1)
        return cls(centers, normals, quaternions, scales, rgb)

    def render(self, viewmats, Ks, width, height, background=None, appearance=None, backend='torch'):
        """Pure 3DGS RGB, expected camera-z depth, world normal and alpha.

        Geometry is frozen, but scales/opacity/color get renderer gradients.
        Normal is alpha-composited anchor normal, not a depth-derived normal.
        """
        values = self.values(appearance)
        rgb = values['rgb']
        features = torch.cat((rgb, self.normals), dim=-1)
        bg = features.new_zeros((viewmats.shape[0], 6))
        if background is not None:
            bg[:, :3] = torch.as_tensor(background, device=bg.device, dtype=bg.dtype)
        if backend == 'torch':
            return rasterize_torch(self.centers, self.quaternions, values['scales'],
                                   values['opacities'], features, viewmats, Ks, width, height, bg)
        from gsplat import rasterization
        rendered, alpha, _ = rasterization(
            means=self.centers, quats=self.quaternions,
            scales=values['scales'], opacities=values['opacities'],
            colors=features, viewmats=viewmats, Ks=Ks, width=int(width), height=int(height),
            backgrounds=bg, render_mode='RGB+ED', packed=True,
            near_plane=0.001, radius_clip=0.0)
        normal = F.normalize(rendered[..., 3:6], dim=-1)
        return {'rgb': rendered[..., :3], 'depth': rendered[..., 6],
                'normal': normal, 'mask': alpha[..., 0]}

    def values(self, appearance=None):
        if appearance is not None:
            return appearance
        return {'rgb': self.color_logits.sigmoid(), 'scales': self.log_scales.exp(),
                'opacities': self.opacity_logits.sigmoid()}

    @torch.no_grad()
    def initialize_colors(self, images, masks, viewmats, Ks, depths=None):
        """Source RGB projection, mask and optional predicted-depth visibility.

        Unobserved anchors keep neutral initialization and are optimized only
        when a legal training view observes them. Query photographs are absent.
        """
        # images [S,H,W,3], masks/depth [S,H,W], fixed OpenCV cameras.
        s, h, w, _ = images.shape
        camera = torch.einsum('sij,nj->sni', viewmats[:, :3, :3], self.centers) + viewmats[:, None, :3, 3]
        projected = torch.einsum('sij,snj->sni', Ks, camera)
        xy = projected[..., :2] / projected[..., 2:3].clamp_min(1e-8)
        grid = torch.stack((2*(xy[..., 0]+0.5)/w-1, 2*(xy[..., 1]+0.5)/h-1), -1)[:, None]
        sampled = F.grid_sample(images.permute(0, 3, 1, 2), grid, align_corners=False)[:, :, 0].permute(0, 2, 1)
        mask = F.grid_sample(masks[:, None].float(), grid, align_corners=False)[:, 0, 0] > 0.5
        valid = mask & (camera[..., 2] > 0) & (grid[:, 0].abs() <= 1).all(-1)
        if depths is not None:
            observed = F.grid_sample(depths[:, None], grid, align_corners=False)[:, 0, 0]
            valid &= (camera[..., 2] - observed).abs() < 0.02 * observed.abs().clamp_min(1e-6)
        weight = valid.float()
        average = (sampled * weight[..., None]).sum(0) / weight.sum(0).clamp_min(1)[:, None]
        rgb = torch.where((weight.sum(0) > 0)[:, None], average, self.color_logits.sigmoid())
        self.color_logits.copy_(torch.logit(rgb.clamp(0.001, 0.999)))

    @torch.no_grad()
    def save_ply(self, path, appearance=None):
        """Standard 3DGS PLY: DC SH, opacity logits, log-scales, wxyz rotation."""
        names = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
                 'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        values = self.values(appearance)
        dc = (values['rgb'] - 0.5) / 0.28209479177387814
        data = torch.cat((self.centers, self.normals, dc, torch.logit(values['opacities'])[:, None],
                          values['scales'].log(), self.quaternions), dim=1).cpu().numpy().astype('<f4')
        header = ['ply', 'format binary_little_endian 1.0', 'comment mesh anchored real 3D Gaussian; all anchors retained',
                  f'element vertex {len(data)}'] + [f'property float {name}' for name in names] + ['end_header']
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as file:
            file.write(('\n'.join(header)+'\n').encode('ascii'))
            file.write(data.tobytes())


class SurfaceAppearance(nn.Module):
    """Shared source-conditioned appearance predictor, saved in model checkpoint.

    Inputs: predicted mesh location/normal and source-visible mean RGB. No target
    image or identity code. The initial function preserves projected source RGB.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 7))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, gaussians, center, scale):
        xyz = (gaussians.centers - torch.as_tensor(center, device=gaussians.centers.device).reshape(1, 3))
        xyz = xyz / torch.as_tensor(scale, device=xyz.device).reshape(1, 1)
        colors = gaussians.color_logits.detach().sigmoid()
        residual = self.mlp(torch.cat((xyz, gaussians.normals, colors), -1))
        return {'rgb': torch.sigmoid(gaussians.color_logits.detach() + residual[:, :3]),
                'scales': torch.exp(gaussians.log_scales.detach() + 0.5 * torch.tanh(residual[:, 3:6])),
                'opacities': torch.sigmoid(gaussians.opacity_logits.detach() + residual[:, 6])}


def quaternion_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def matrix_quaternion(matrix):
    """Largest-component conversion, avoiding optional scipy/BLAS dependency."""
    m = matrix
    roots = torch.stack((1+m[:,0,0]+m[:,1,1]+m[:,2,2],
                         1+m[:,0,0]-m[:,1,1]-m[:,2,2],
                         1-m[:,0,0]+m[:,1,1]-m[:,2,2],
                         1-m[:,0,0]-m[:,1,1]+m[:,2,2]), -1).clamp_min(0).sqrt()
    candidates = torch.stack((
        torch.stack((roots[:,0].square(),m[:,2,1]-m[:,1,2],m[:,0,2]-m[:,2,0],m[:,1,0]-m[:,0,1]),-1),
        torch.stack((m[:,2,1]-m[:,1,2],roots[:,1].square(),m[:,1,0]+m[:,0,1],m[:,0,2]+m[:,2,0]),-1),
        torch.stack((m[:,0,2]-m[:,2,0],m[:,1,0]+m[:,0,1],roots[:,2].square(),m[:,2,1]+m[:,1,2]),-1),
        torch.stack((m[:,1,0]-m[:,0,1],m[:,2,0]+m[:,0,2],m[:,2,1]+m[:,1,2],roots[:,3].square()),-1)), 1)
    candidates = candidates / (2*roots[...,None].clamp_min(0.1))
    selected = candidates[torch.arange(len(m)), roots.argmax(-1)]
    return F.normalize(selected,dim=-1)


def rasterize_torch(means, quats, scales, opacities, features, viewmats, Ks, width, height, backgrounds, tile=16):
    """Native Torch EWA 3DGS: full covariance + depth-sorted alpha composition.

    Reference implementation for local platforms without CUDA GS extensions.
    Tiles cull only outside the Gaussian's conventional 3-sigma screen support;
    all Gaussian parameters are retained in export. Slower than gsplat.
    """
    rotations = quaternion_matrix(quats)
    cov_world = (rotations * scales[:, None, :].square()) @ rotations.transpose(-1, -2)
    images, depths, normals, masks = [], [], [], []
    for camera, intrinsic, background in zip(viewmats, Ks, backgrounds):
        rotation, translation = camera[:3, :3], camera[:3, 3]
        xyz = means @ rotation.T + translation
        z = xyz[:, 2]
        safe_z = z.clamp_min(1e-6)
        x, y = xyz[:, 0], xyz[:, 1]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        projected = xyz @ intrinsic.T
        uv = projected[:, :2] / safe_z[:, None]
        zero = torch.zeros_like(z)
        jacobian = torch.stack((fx/safe_z, zero, -fx*x/safe_z.square(),
                                zero, fy/safe_z, -fy*y/safe_z.square()), -1).reshape(-1, 2, 3)
        camera_cov = rotation[None] @ cov_world @ rotation.T[None]
        screen_cov = jacobian @ camera_cov @ jacobian.transpose(-1, -2)
        screen_cov = screen_cov + torch.eye(2, device=z.device, dtype=z.dtype)[None] * 0.3
        inverse = torch.linalg.inv(screen_cov)
        radius = 3 * torch.linalg.eigvalsh(screen_cov).amax(-1).sqrt()
        rows = []
        for top in range(0, height, tile):
            columns = []
            for left in range(0, width, tile):
                th, tw = min(tile, height-top), min(tile, width-left)
                # Discrete visibility/sort are standard rasterizer boundaries.
                visible = ((z > 0.001) & (uv[:, 0]+radius >= left) & (uv[:, 0]-radius < left+tw)
                           & (uv[:, 1]+radius >= top) & (uv[:, 1]-radius < top+th))
                indices = torch.nonzero(visible, as_tuple=False).flatten()
                indices = indices[torch.argsort(z[indices])]
                yy, xx = torch.meshgrid(torch.arange(top, top+th, device=z.device, dtype=z.dtype),
                                        torch.arange(left, left+tw, device=z.device, dtype=z.dtype), indexing='ij')
                # Match this model's backproject: integer-index pixel centers.
                pixels = torch.stack((xx, yy), -1).reshape(-1, 2)
                delta = pixels[None] - uv[indices, None]
                exponent = -0.5 * torch.einsum('npi,nij,npj->np', delta, inverse[indices], delta)
                alpha = (opacities[indices, None] * torch.exp(exponent)).clamp(max=0.99)
                alpha = torch.where(exponent >= -4.5, alpha, torch.zeros_like(alpha))
                trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha), 0), 0)
                if len(indices):
                    weights = trans[:-1] * alpha
                    acc = weights.sum(0)
                    color = weights.T @ features[indices] + trans[-1, :, None] * background
                    depth = (weights * z[indices, None]).sum(0) / acc.clamp_min(1e-8)
                else:
                    acc = torch.zeros(th*tw, device=z.device, dtype=z.dtype)
                    color = background.expand(th*tw, -1)
                    depth = acc
                columns.append(torch.cat((color, depth[:, None], acc[:, None]), -1).reshape(th, tw, -1))
            rows.append(torch.cat(columns, 1))
        frame = torch.cat(rows, 0)
        images.append(frame[..., :3])
        normals.append(F.normalize(frame[..., 3:6], dim=-1))
        depths.append(frame[..., 6])
        masks.append(frame[..., 7])
    return {'rgb': torch.stack(images), 'depth': torch.stack(depths),
            'normal': torch.stack(normals), 'mask': torch.stack(masks)}


def train_appearance(gaussians, images, masks, viewmats, Ks, steps=200, lr=0.01, log_callback=None):
    """Per-scene appearance fitting on explicitly supplied legal source/train views.

    Images are [S,H,W,3]. No geometry/point displacement parameters exist.
    Loss means use S*H*W*3 RGB entries and S*H*W alpha entries.
    """
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    h, w = images.shape[1:3]
    history = []
    for step in range(1, steps+1):
        # A single rotating training camera bounds renderer memory.
        index = (step-1) % len(images)
        prediction = gaussians.render(viewmats[index:index+1], Ks[index:index+1], w, h)
        target = images[index:index+1] * masks[index:index+1, ..., None]
        rgb_loss = (prediction['rgb'] - target).abs().mean()
        mask_loss = (prediction['mask'] - masks[index:index+1]).square().mean()
        loss = rgb_loss + 0.1 * mask_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Gaussian appearance loss became nonfinite at update {step}.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {'update': step, 'loss': float(loss.detach()), 'rgb_l1': float(rgb_loss.detach()), 'mask_mse': float(mask_loss.detach())}
        history.append(row)
        if log_callback is not None:
            log_callback(row)
    return history
