"""Shared oriented-point geometry decoder; SAP DPSR convention, not an SDF.

Spectral formulation adapted from autonomousvision/shape_as_points (MIT).
See ../third_party/SAP_LICENSE.txt. Marching cubes is export-only.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _neighbors(points, resolution):
    position = points * resolution
    lower = position.floor()
    fraction = position - lower
    for corner in itertools.product((0, 1), repeat=3):
        offset = points.new_tensor(corner)
        index = (lower.long() + offset.long()) % resolution
        weight = torch.where(offset.bool(), fraction, 1 - fraction).prod(-1)
        flat = (index[..., 0] * resolution + index[..., 1]) * resolution + index[..., 2]
        yield flat, weight


def point_rasterize(points, values, resolution):
    """Periodic trilinear splat [B,N,3], [B,N,C] -> [B,C,R,R,R]."""
    b, _, c = values.shape
    grid = values.new_zeros(b, c, resolution ** 3)
    for index, weight in _neighbors(points, resolution):
        grid = grid.scatter_add(2, index[:, None].expand(-1, c, -1),
                                (values * weight[..., None]).transpose(1, 2))
    return grid.reshape(b, c, resolution, resolution, resolution)


def grid_interp(grid, points):
    """Periodic trilinear interpolation [B,R,R,R] at [B,N,3]."""
    flat = grid.flatten(1)
    value = points.new_zeros(points.shape[:2])
    for index, weight in _neighbors(points, grid.shape[-1]):
        value = value + flat.gather(1, index) * weight
    return value


class DPSR(nn.Module):
    """SAP spectral Poisson solver with surface mean shift and corner scaling.

    Coordinates use the periodic [0,1)^3 grid, samples at i/R. Outward normals
    produce negative interior and positive exterior under SAP's final sign.
    The input object must fit strictly inside the cube.
    """
    def __init__(self, resolution=128, sigma=2.0):
        super().__init__()
        self.resolution = int(resolution)
        self.sigma = float(sigma)
        r = self.resolution
        freq = [torch.fft.fftfreq(r, d=1/r), torch.fft.fftfreq(r, d=1/r),
                torch.fft.rfftfreq(r, d=1/r)]
        frequency = torch.stack(torch.meshgrid(*freq, indexing='ij'), -1)
        self.register_buffer('frequency', frequency)
        self.register_buffer('smoothing', torch.exp(-0.5 * (self.sigma * 2 * frequency.norm(dim=-1) / r) ** 2))

    def forward(self, points, normals):
        r = self.resolution
        vector_grid = point_rasterize(points, normals, r)
        spectrum = torch.fft.rfftn(vector_grid, dim=(-3, -2, -1))
        omega = self.frequency.to(points) * (2 * math.pi)
        # SAP: DivN = -i * sum(omega*N), Lap = -sum(omega^2).
        divergence = (-1j * spectrum * omega.permute(3, 0, 1, 2)[None]).sum(1)
        laplacian = -omega.square().sum(-1)
        field_spectrum = divergence * self.smoothing.to(points) / (laplacian + 1e-6)
        dc_mask = torch.ones_like(laplacian)
        dc_mask[0, 0, 0] = 0
        field = torch.fft.irfftn(field_spectrum * dc_mask, s=(r, r, r), dim=(-3, -2, -1))
        field = field - grid_interp(field, points).mean(1)[:, None, None, None]
        corner = field[:, 0, 0, 0].abs()
        # A vanishing corner means there is no usable enclosing indicator.
        if bool((corner.detach() < 1e-12).any()):
            raise ValueError('DPSR corner normalization vanished; check points/normals and input cube.')
        return -0.5 * field / corner[:, None, None, None]


def world_to_unit(points, center, scale):
    center = torch.as_tensor(center, dtype=points.dtype, device=points.device).reshape(-1, 1, 3)
    scale = torch.as_tensor(scale, dtype=points.dtype, device=points.device).reshape(-1, 1, 1)
    return (points - center) / scale + 0.5


class GeometryDecoder(nn.Module):
    def __init__(self, resolution=128, sigma=2.0, channels=8):
        super().__init__()
        self.dpsr = DPSR(resolution, sigma)
        # Low-resolution shared context reaches occluded areas without per-person grids.
        self.refine = nn.Sequential(nn.Conv3d(1, channels, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(channels, channels, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(channels, 1, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, points, normals, center=None, scale=None, unit_transform=None):
        """真实有向点先检非有限和原cube，绝不裁点或更改尺度。"""
        if not bool(torch.isfinite(points).all() and torch.isfinite(normals).all()):
            raise ValueError('Nonfinite actual DPSR points or normals; preserve original input and stop.')
        if unit_transform is not None:
            center, scale = unit_transform['center'], unit_transform['scale']
        unit = world_to_unit(points, center, scale)
        if bool(((unit.detach() <= 0) | (unit.detach() >= 1)).any()):
            raise ValueError('Source points outside input-defined DPSR cube; do not silently clip or use scan bbox.')
        base = self.dpsr(unit, F.normalize(normals, dim=-1))
        low = F.avg_pool3d(base[:, None], 4)
        residual = F.interpolate(self.refine(low), size=base.shape[-3:], mode='trilinear', align_corners=False)[:, 0]
        return {'indicator': base + residual, 'indicator_base': base, 'unit_points': unit}

    @torch.no_grad()
    def target(self, scan_points, scan_normals, center, scale):
        """Training/oracle only. Never call with scan in the inference path."""
        unit = world_to_unit(scan_points, center, scale)
        if bool(((unit <= 0) | (unit >= 1)).any()):
            raise ValueError('Legal training scan exceeds the INPUT-defined cube; fix input normalization.')
        return self.dpsr(unit, F.normalize(scan_normals, dim=-1))


def export_mesh(indicator, center, scale, path=None):
    """Ordinary CPU marching cubes. Detached, all components preserved."""
    from skimage.measure import marching_cubes
    field = indicator.detach().float().cpu().numpy() if torch.is_tensor(indicator) else np.asarray(indicator)
    field = np.squeeze(field)
    vertices, faces, _, _ = marching_cubes(field, level=0, gradient_direction='descent')
    # Grid samples live at i/R (not i/(R-1)); see SAP's real_scale=False.
    vertices = (vertices / field.shape[0] - 0.5) * float(np.asarray(scale).reshape(-1)[0]) + np.asarray(center).reshape(3)
    # SAP negative-inside field with skimage descent gives outward face winding.
    if path is not None:
        import trimesh
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices, faces, process=False).export(str(path))
    return vertices.astype(np.float32), faces.astype(np.int64)
