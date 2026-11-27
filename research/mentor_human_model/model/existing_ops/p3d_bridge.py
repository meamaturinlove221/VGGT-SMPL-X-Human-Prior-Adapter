"""Cached fitted-mesh surface raster to dense VGGT patch features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _validate_camera(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, batch: int) -> int:
    if K.ndim != 4 or K.shape[0] != batch or K.shape[-2:] != (3, 3):
        raise ValueError("K must have shape [B,S,3,3]")
    if R.shape != K.shape or t.shape != (*K.shape[:2], 3):
        raise ValueError("R/t must have shapes [B,S,3,3]/[B,S,3]")
    if K.device != R.device or K.device != t.device or K.dtype != R.dtype or K.dtype != t.dtype:
        raise ValueError("K/R/t must share device and dtype")
    if not (torch.isfinite(K).all() and torch.isfinite(R).all() and torch.isfinite(t).all()):
        raise ValueError("camera arrays must be finite")
    if bool((K[..., 0, 0] <= 0).any() or (K[..., 1, 1] <= 0).any()):
        raise ValueError("camera focal lengths must be positive")
    return int(K.shape[1])


def _batched_faces(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.ndim not in (2, 3) or faces.shape[-1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be int64 [F,3] or [B,F,3]")
    batched = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1) if faces.ndim == 2 else faces
    if batched.shape[0] != vertices.shape[0] or batched.device != vertices.device:
        raise ValueError("faces must share vertex batch and device")
    if batched.numel() and (int(batched.min()) < 0 or int(batched.max()) >= vertices.shape[1]):
        raise ValueError("faces contain an out-of-range vertex index")
    return batched


@dataclass
class DenseMeshRasterCache:
    """Immutable-lineage raster cache; feature tensors remain outside the cache."""

    batch: int
    views: int
    points: int
    face_count: int
    image_height: int
    image_width: int
    patch_size: int
    patch_grid_height: int
    patch_grid_width: int
    visible_batch_index: torch.Tensor
    visible_view_index: torch.Tensor
    visible_pixel_linear_index: torch.Tensor
    visible_vertex_indices: torch.Tensor
    visible_barycentric: torch.Tensor
    visible_patch_index: torch.Tensor
    patch_pixel_counts: torch.Tensor
    silhouette: torch.Tensor
    visible_face_index: torch.Tensor
    barycentric_map: torch.Tensor
    visible_depth: torch.Tensor
    region_masks: dict[str, torch.Tensor] | None
    region_binding_status: str
    vertices_snapshot: torch.Tensor = field(repr=False)
    faces_snapshot: torch.Tensor = field(repr=False)
    K_snapshot: torch.Tensor = field(repr=False)
    R_snapshot: torch.Tensor = field(repr=False)
    t_snapshot: torch.Tensor = field(repr=False)
    vertices_object_id: int
    faces_object_id: int
    K_object_id: int
    R_object_id: int
    t_object_id: int
    vertices_version: int
    faces_version: int
    K_version: int
    R_version: int
    t_version: int
    rasterization_count: int = 1
    schema: str = "stage47.dense_mesh_raster_cache.v1"

    @property
    def patch_count(self) -> int:
        return self.patch_grid_height * self.patch_grid_width

    def validate_lineage(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> None:
        current = (vertices, faces, K, R, t)
        expected_ids = (self.vertices_object_id, self.faces_object_id, self.K_object_id, self.R_object_id, self.t_object_id)
        expected_versions = (self.vertices_version, self.faces_version, self.K_version, self.R_version, self.t_version)
        labels = ("vertices", "faces", "K", "R", "t")
        snapshots = (self.vertices_snapshot, self.faces_snapshot, self.K_snapshot, self.R_snapshot, self.t_snapshot)
        for label, value, object_id, version, snapshot in zip(labels, current, expected_ids, expected_versions, snapshots):
            if id(value) != object_id:
                raise ValueError(f"dense raster cache {label} object identity mismatch")
            if value._version != version:
                raise ValueError(f"dense raster cache {label} in-place version mismatch")
            if value.shape != snapshot.shape or value.dtype != snapshot.dtype or value.device != snapshot.device:
                raise ValueError(f"dense raster cache {label} shape/device/dtype mismatch")
            if not torch.equal(value.detach(), snapshot):
                raise ValueError(f"dense raster cache {label} ordered values mismatch")

    def lineage(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mesh_shape": list(self.vertices_snapshot.shape),
            "faces_shape": list(self.faces_snapshot.shape),
            "view_count": self.views,
            "image_shape": [self.image_height, self.image_width],
            "patch_grid": [self.patch_grid_height, self.patch_grid_width],
            "geometry_device": str(self.vertices_snapshot.device),
            "geometry_dtype": str(self.vertices_snapshot.dtype),
            "camera_device": str(self.K_snapshot.device),
            "camera_dtype": str(self.K_snapshot.dtype),
            "mesh_order_exact": True,
            "reference_krt_exact": True,
            "rasterization_count": self.rasterization_count,
            "region_binding_status": self.region_binding_status,
        }


def _validate_regions(
    face_region_ids: torch.Tensor | None,
    region_names: tuple[str, ...] | None,
    batched_faces: torch.Tensor,
) -> tuple[torch.Tensor | None, str]:
    if face_region_ids is None:
        if region_names is not None:
            raise ValueError("face_region_ids are required with region_names")
        return None, "UNAVAILABLE_FACE_REGION_BINDING_NOT_PROVEN"
    if face_region_ids.ndim == 1:
        face_region_ids = face_region_ids.unsqueeze(0).expand(batched_faces.shape[0], -1)
    if face_region_ids.shape != batched_faces.shape[:2] or face_region_ids.dtype != torch.long:
        raise ValueError("face_region_ids must be int64 [F] or [B,F]")
    if face_region_ids.device != batched_faces.device:
        raise ValueError("face_region_ids must share mesh device")
    if region_names is None or not region_names:
        raise ValueError("region_names are required with face_region_ids")
    if len(set(region_names)) != len(region_names):
        raise ValueError("region_names must be unique")
    if face_region_ids.numel() and (int(face_region_ids.min()) < 0 or int(face_region_ids.max()) >= len(region_names)):
        raise ValueError("face_region_ids contain an invalid region")
    return face_region_ids, "CALLER_SUPPLIED_FACE_REGION_BINDING_VALIDATED_NOT_AUTHORED_BY_BRIDGE"


@torch.no_grad()
def prepare_dense_mesh_raster_cache(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reference_K: torch.Tensor,
    reference_R: torch.Tensor,
    reference_t: torch.Tensor,
    *,
    image_height: int = 518,
    image_width: int = 518,
    patch_size: int = 14,
    face_region_ids: torch.Tensor | None = None,
    vertex_region_ids: torch.Tensor | None = None,
    region_names: tuple[str, ...] | None = None,
    region_binding_label: str | None = None,
) -> DenseMeshRasterCache:
    """Rasterize fixed mesh/reference KRT exactly once without an autograd graph."""
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not torch.is_floating_point(vertices):
        raise ValueError("vertices must be floating [B,N,3]")
    if not torch.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if image_height <= 0 or image_width <= 0 or patch_size <= 0:
        raise ValueError("image dimensions and patch_size must be positive")
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    batch, points, _ = vertices.shape
    batched_faces = _batched_faces(vertices, faces)
    views = _validate_camera(reference_K, reference_R, reference_t, batch)
    if reference_K.device != vertices.device or reference_K.dtype != vertices.dtype:
        raise ValueError("reference camera must share mesh device and dtype")
    if face_region_ids is not None and vertex_region_ids is not None:
        raise ValueError("face_region_ids and vertex_region_ids are mutually exclusive")
    face_region_ids, region_status = _validate_regions(
        face_region_ids,
        region_names if face_region_ids is not None else None,
        batched_faces,
    )
    if vertex_region_ids is not None:
        if vertex_region_ids.ndim == 1:
            vertex_region_ids = vertex_region_ids.unsqueeze(0).expand(batch, -1)
        if vertex_region_ids.shape != vertices.shape[:2] or vertex_region_ids.dtype != torch.long:
            raise ValueError("vertex_region_ids must be int64 [N] or [B,N]")
        if vertex_region_ids.device != vertices.device or region_names is None or not region_names:
            raise ValueError("vertex_region_ids require same device and nonempty region_names")
        if vertex_region_ids.numel() and (int(vertex_region_ids.min()) < 0 or int(vertex_region_ids.max()) >= len(region_names)):
            raise ValueError("vertex_region_ids contain an invalid region")
        if region_binding_label != "HISTORICAL_DOMINANT_LBS_DIAGNOSTIC_PROXY_EXACT_TOPOLOGY_BOUND":
            raise ValueError("vertex region proxy requires exact historical diagnostic binding label")
        region_status = region_binding_label

    face_indices = torch.full((batch, views, image_height, image_width), -1, dtype=torch.long, device=vertices.device)
    barycentric = vertices.new_zeros(batch, views, image_height, image_width, 3)
    depths = vertices.new_full((batch, views, image_height, image_width), float("inf"))
    camera_xyz = torch.einsum("bsij,bnj->bsni", reference_R, vertices) + reference_t[:, :, None]
    z_all = camera_xyz[..., 2]
    eps = torch.finfo(vertices.dtype).eps
    u_all = reference_K[..., 0, 0, None] * camera_xyz[..., 0] / z_all.clamp_min(eps) + reference_K[..., 0, 2, None]
    v_all = reference_K[..., 1, 1, None] * camera_xyz[..., 1] / z_all.clamp_min(eps) + reference_K[..., 1, 2, None]
    for b in range(batch):
        for s in range(views):
            for face_id, tri in enumerate(batched_faces[b]):
                tri_z = z_all[b, s, tri]
                if bool((tri_z <= 0).any()):
                    continue
                tri_u, tri_v = u_all[b, s, tri], v_all[b, s, tri]
                x0, x1 = max(0, int(torch.floor(tri_u.min()))), min(image_width - 1, int(torch.ceil(tri_u.max())))
                y0, y1 = max(0, int(torch.floor(tri_v.min()))), min(image_height - 1, int(torch.ceil(tri_v.max())))
                if x0 > x1 or y0 > y1:
                    continue
                ys = torch.arange(y0, y1 + 1, device=vertices.device, dtype=vertices.dtype)
                xs = torch.arange(x0, x1 + 1, device=vertices.device, dtype=vertices.dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                denom = (tri_v[1] - tri_v[2]) * (tri_u[0] - tri_u[2]) + (tri_u[2] - tri_u[1]) * (tri_v[0] - tri_v[2])
                if bool(denom.abs() <= eps):
                    continue
                l0 = ((tri_v[1] - tri_v[2]) * (xx - tri_u[2]) + (tri_u[2] - tri_u[1]) * (yy - tri_v[2])) / denom
                l1 = ((tri_v[2] - tri_v[0]) * (xx - tri_u[2]) + (tri_u[0] - tri_u[2]) * (yy - tri_v[2])) / denom
                screen = torch.stack((l0, l1, 1.0 - l0 - l1), dim=-1)
                inside = (screen >= -8 * eps).all(dim=-1)
                reciprocal_depth = (screen / tri_z).sum(dim=-1)
                candidate_depth = reciprocal_depth.reciprocal()
                current = depths[b, s, y0 : y1 + 1, x0 : x1 + 1]
                update = inside & torch.isfinite(candidate_depth) & (candidate_depth < current)
                if not bool(update.any()):
                    continue
                perspective = (screen / tri_z) * candidate_depth[..., None]
                current[update] = candidate_depth[update]
                face_indices[b, s, y0 : y1 + 1, x0 : x1 + 1][update] = face_id
                barycentric[b, s, y0 : y1 + 1, x0 : x1 + 1][update] = perspective[update]

    silhouette = face_indices >= 0
    visible = silhouette.nonzero(as_tuple=False)
    visible_batch = visible[:, 0]
    visible_view = visible[:, 1]
    visible_y, visible_x = visible[:, 2], visible[:, 3]
    visible_faces = face_indices[silhouette]
    visible_vertices = batched_faces[visible_batch, visible_faces]
    visible_bary = barycentric[silhouette]
    pixel_linear = visible_y * image_width + visible_x
    grid_w = image_width // patch_size
    patch = (visible_y // patch_size) * grid_w + (visible_x // patch_size)
    patch_count = (image_height // patch_size) * grid_w
    global_patch = (visible_batch * views + visible_view) * patch_count + patch
    counts = torch.bincount(global_patch, minlength=batch * views * patch_count).reshape(batch, views, patch_count)
    counts = counts.to(dtype=vertices.dtype)
    region_masks = None
    if (face_region_ids is not None or vertex_region_ids is not None) and region_names is not None:
        region_masks = {}
        if vertex_region_ids is not None:
            dominant_corner = visible_bary.argmax(dim=-1)
            dominant_vertex = visible_vertices.gather(1, dominant_corner[:, None]).squeeze(1)
            visible_region = vertex_region_ids[visible_batch, dominant_vertex]
        else:
            assert face_region_ids is not None
            visible_region = face_region_ids[visible_batch, visible_faces]
        for region_id, name in enumerate(region_names):
            mask = torch.zeros_like(silhouette)
            mask[silhouette] = visible_region == region_id
            region_masks[name] = mask
    depths = torch.where(silhouette, depths, torch.zeros_like(depths))
    return DenseMeshRasterCache(
        batch=batch, views=views, points=points, face_count=int(batched_faces.shape[1]),
        image_height=image_height, image_width=image_width, patch_size=patch_size,
        patch_grid_height=image_height // patch_size, patch_grid_width=grid_w,
        visible_batch_index=visible_batch, visible_view_index=visible_view,
        visible_pixel_linear_index=pixel_linear, visible_vertex_indices=visible_vertices,
        visible_barycentric=visible_bary, visible_patch_index=patch,
        patch_pixel_counts=counts, silhouette=silhouette, visible_face_index=face_indices,
        barycentric_map=barycentric,
        visible_depth=depths, region_masks=region_masks, region_binding_status=region_status,
        vertices_snapshot=vertices.detach().clone(), faces_snapshot=faces.detach().clone(),
        K_snapshot=reference_K.detach().clone(), R_snapshot=reference_R.detach().clone(), t_snapshot=reference_t.detach().clone(),
        vertices_object_id=id(vertices), faces_object_id=id(faces), K_object_id=id(reference_K),
        R_object_id=id(reference_R), t_object_id=id(reference_t),
        vertices_version=vertices._version, faces_version=faces._version, K_version=reference_K._version,
        R_version=reference_R._version, t_version=reference_t._version,
    )


@torch.no_grad()
def prepare_dense_mesh_raster_cache_vectorized(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reference_K: torch.Tensor,
    reference_R: torch.Tensor,
    reference_t: torch.Tensor,
    *,
    image_height: int = 518,
    image_width: int = 518,
    patch_size: int = 14,
    face_chunk_size: int = 32,
) -> DenseMeshRasterCache:
    """Rasterize a detached dynamic mesh without per-face GPU scalar sync.

    This is the same pixel-center, perspective-correct, strict-nearer z-buffer
    contract as :func:`prepare_dense_mesh_raster_cache`.  It evaluates face
    chunks over the image grid and keeps original face order for exact-depth
    ties.  The bounded implementation is used where the mesh changes every
    training forward; the fixed-mesh dense bridge continues to use its original
    cache producer.
    """

    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not torch.is_floating_point(vertices):
        raise ValueError("vertices must be floating [B,N,3]")
    if not torch.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if image_height <= 0 or image_width <= 0 or patch_size <= 0:
        raise ValueError("image dimensions and patch_size must be positive")
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    if face_chunk_size <= 0:
        raise ValueError("face_chunk_size must be positive")
    batch, points, _ = vertices.shape
    batched_faces = _batched_faces(vertices, faces)
    views = _validate_camera(reference_K, reference_R, reference_t, batch)
    if reference_K.device != vertices.device or reference_K.dtype != vertices.dtype:
        raise ValueError("reference camera must share mesh device and dtype")

    face_indices = torch.full(
        (batch, views, image_height, image_width), -1, dtype=torch.long, device=vertices.device
    )
    barycentric = vertices.new_zeros(batch, views, image_height, image_width, 3)
    depths = vertices.new_full((batch, views, image_height, image_width), float("inf"))
    camera_xyz = torch.einsum("bsij,bnj->bsni", reference_R, vertices) + reference_t[:, :, None]
    z_all = camera_xyz[..., 2]
    eps = torch.finfo(vertices.dtype).eps
    u_all = reference_K[..., 0, 0, None] * camera_xyz[..., 0] / z_all.clamp_min(eps) + reference_K[..., 0, 2, None]
    v_all = reference_K[..., 1, 1, None] * camera_xyz[..., 1] / z_all.clamp_min(eps) + reference_K[..., 1, 2, None]
    ys = torch.arange(image_height, device=vertices.device, dtype=vertices.dtype)
    xs = torch.arange(image_width, device=vertices.device, dtype=vertices.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    for b in range(batch):
        for s in range(views):
            current_depth = depths[b, s]
            current_face = face_indices[b, s]
            current_bary = barycentric[b, s]
            for start in range(0, int(batched_faces.shape[1]), face_chunk_size):
                stop = min(start + face_chunk_size, int(batched_faces.shape[1]))
                tri = batched_faces[b, start:stop]
                tri_z = z_all[b, s, tri]
                tri_u = u_all[b, s, tri]
                tri_v = v_all[b, s, tri]
                denom = (
                    (tri_v[:, 1] - tri_v[:, 2]) * (tri_u[:, 0] - tri_u[:, 2])
                    + (tri_u[:, 2] - tri_u[:, 1]) * (tri_v[:, 0] - tri_v[:, 2])
                )
                valid_face = (tri_z > 0).all(dim=-1) & (denom.abs() > eps)
                safe_denom = torch.where(valid_face, denom, torch.ones_like(denom))
                l0 = (
                    (tri_v[:, 1] - tri_v[:, 2])[:, None, None] * (xx - tri_u[:, 2, None, None])
                    + (tri_u[:, 2] - tri_u[:, 1])[:, None, None] * (yy - tri_v[:, 2, None, None])
                ) / safe_denom[:, None, None]
                l1 = (
                    (tri_v[:, 2] - tri_v[:, 0])[:, None, None] * (xx - tri_u[:, 2, None, None])
                    + (tri_u[:, 0] - tri_u[:, 2])[:, None, None] * (yy - tri_v[:, 2, None, None])
                ) / safe_denom[:, None, None]
                screen = torch.stack((l0, l1, 1.0 - l0 - l1), dim=-1)
                inside = (screen >= -8 * eps).all(dim=-1) & valid_face[:, None, None]
                reciprocal_depth = (screen / tri_z[:, None, None, :].clamp_min(eps)).sum(dim=-1)
                candidate_depth = reciprocal_depth.reciprocal()
                candidate_depth = torch.where(
                    inside & torch.isfinite(candidate_depth),
                    candidate_depth,
                    torch.full_like(candidate_depth, float("inf")),
                )
                chunk_depth, local_face = candidate_depth.min(dim=0)
                update = chunk_depth < current_depth
                gather_screen = local_face[None, ..., None].expand(1, image_height, image_width, 3)
                winning_screen = screen.gather(0, gather_screen).squeeze(0)
                gather_z = local_face[:, :, None].expand(image_height, image_width, 3)
                winning_z = tri_z.gather(0, gather_z.reshape(-1, 3)).reshape(image_height, image_width, 3)
                winning_bary = (winning_screen / winning_z) * chunk_depth[..., None]
                current_depth = torch.where(update, chunk_depth, current_depth)
                current_face = torch.where(update, local_face + start, current_face)
                current_bary = torch.where(update[..., None], winning_bary, current_bary)
            depths[b, s] = current_depth
            face_indices[b, s] = current_face
            barycentric[b, s] = current_bary

    silhouette = face_indices >= 0
    visible = silhouette.nonzero(as_tuple=False)
    visible_batch = visible[:, 0]
    visible_view = visible[:, 1]
    visible_y, visible_x = visible[:, 2], visible[:, 3]
    visible_faces = face_indices[silhouette]
    visible_vertices = batched_faces[visible_batch, visible_faces]
    visible_bary = barycentric[silhouette]
    pixel_linear = visible_y * image_width + visible_x
    grid_w = image_width // patch_size
    patch = (visible_y // patch_size) * grid_w + (visible_x // patch_size)
    patch_count = (image_height // patch_size) * grid_w
    global_patch = (visible_batch * views + visible_view) * patch_count + patch
    counts = torch.bincount(global_patch, minlength=batch * views * patch_count).reshape(batch, views, patch_count)
    counts = counts.to(dtype=vertices.dtype)
    depths = torch.where(silhouette, depths, torch.zeros_like(depths))
    return DenseMeshRasterCache(
        batch=batch, views=views, points=points, face_count=int(batched_faces.shape[1]),
        image_height=image_height, image_width=image_width, patch_size=patch_size,
        patch_grid_height=image_height // patch_size, patch_grid_width=grid_w,
        visible_batch_index=visible_batch, visible_view_index=visible_view,
        visible_pixel_linear_index=pixel_linear, visible_vertex_indices=visible_vertices,
        visible_barycentric=visible_bary, visible_patch_index=patch,
        patch_pixel_counts=counts, silhouette=silhouette, visible_face_index=face_indices,
        barycentric_map=barycentric, visible_depth=depths, region_masks=None,
        region_binding_status="UNAVAILABLE_FACE_REGION_BINDING_NOT_PROVEN",
        vertices_snapshot=vertices.detach().clone(), faces_snapshot=faces.detach().clone(),
        K_snapshot=reference_K.detach().clone(), R_snapshot=reference_R.detach().clone(),
        t_snapshot=reference_t.detach().clone(), vertices_object_id=id(vertices),
        faces_object_id=id(faces), K_object_id=id(reference_K), R_object_id=id(reference_R),
        t_object_id=id(reference_t), vertices_version=vertices._version,
        faces_version=faces._version, K_version=reference_K._version,
        R_version=reference_R._version, t_version=reference_t._version,
    )


def apply_dense_mesh_raster_cache(
    encoded_vertex_features: torch.Tensor,
    cache: DenseMeshRasterCache,
    *,
    token_bridge: torch.nn.Module | None = None,
    return_dense_features: bool = False,
) -> dict[str, Any]:
    """Vectorized differentiable feature gather/scatter; never re-rasterizes."""
    if encoded_vertex_features.ndim != 3 or encoded_vertex_features.shape[:2] != (cache.batch, cache.points):
        raise ValueError("encoded_vertex_features shape does not match raster cache")
    if not torch.is_floating_point(encoded_vertex_features) or not torch.isfinite(encoded_vertex_features).all():
        raise ValueError("encoded_vertex_features must be finite floating")
    if encoded_vertex_features.device != cache.visible_vertex_indices.device:
        raise ValueError("encoded_vertex_features device does not match raster cache")
    tri_features = encoded_vertex_features[
        cache.visible_batch_index[:, None], cache.visible_vertex_indices
    ]
    visible_features = (
        tri_features * cache.visible_barycentric.to(encoded_vertex_features.dtype)[..., None]
    ).sum(dim=1)
    channels = int(encoded_vertex_features.shape[-1])
    global_patch = (
        (cache.visible_batch_index * cache.views + cache.visible_view_index) * cache.patch_count
        + cache.visible_patch_index
    )
    sums = encoded_vertex_features.new_zeros(cache.batch * cache.views * cache.patch_count, channels)
    sums.index_add_(0, global_patch, visible_features)
    patch_features = sums.reshape(cache.batch, cache.views, cache.patch_count, channels)
    patch_features = patch_features / cache.patch_pixel_counts.to(encoded_vertex_features.dtype)[..., None].clamp_min(1.0)
    patch_tokens = token_bridge(patch_features) if token_bridge is not None else patch_features
    if not torch.isfinite(patch_tokens).all():
        raise ValueError("dense mesh cache apply produced non-finite patch tokens")
    result: dict[str, Any] = {
        "mentor_dual_xf_prior_tokens": patch_tokens,
        "dense_patch_features": patch_features,
        "visible_pixel_features": visible_features,
        "patch_pixel_counts": cache.patch_pixel_counts,
        "patch_occupancy": cache.patch_pixel_counts > 0,
        "silhouette": cache.silhouette,
        "visible_depth": cache.visible_depth,
        "visible_face_index": cache.visible_face_index,
        "visible_barycentric": cache.barycentric_map,
        "packed_visible_barycentric": cache.visible_barycentric,
        "region_masks": cache.region_masks,
        "region_binding_status": cache.region_binding_status,
        "rasterization_count": cache.rasterization_count,
        "prior_projection_camera_role": "REFERENCE_KRT_PRIOR_RASTER_ONLY_NEVER_OUTPUT_CAMERA",
        "raster_cache_lineage": cache.lineage(),
        "raster_cache_object_id": id(cache),
    }
    if return_dense_features:
        dense = encoded_vertex_features.new_zeros(
            cache.batch, cache.views, cache.image_height * cache.image_width, channels
        )
        dense[cache.visible_batch_index, cache.visible_view_index, cache.visible_pixel_linear_index] = visible_features
        result["dense_visible_features"] = dense.reshape(
            cache.batch, cache.views, cache.image_height, cache.image_width, channels
        )
    return result


def rasterize_encoded_mesh_to_dense_p3d(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    encoded_vertex_features: torch.Tensor,
    prior_projection_K: torch.Tensor,
    prior_projection_R: torch.Tensor,
    prior_projection_t: torch.Tensor,
    *,
    image_size: int = 518,
    patch_size: int = 14,
    token_bridge: torch.nn.Module | None = None,
    face_region_ids: torch.Tensor | None = None,
    vertex_region_ids: torch.Tensor | None = None,
    region_names: tuple[str, ...] | None = None,
    region_binding_label: str | None = None,
    canonical_output_K: torch.Tensor | None = None,
    canonical_output_R: torch.Tensor | None = None,
    canonical_output_t: torch.Tensor | None = None,
    return_dense_features: bool = False,
) -> dict[str, Any]:
    """Explicit uncached compatibility path for small tests only."""
    supplied = (canonical_output_K is not None, canonical_output_R is not None, canonical_output_t is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("canonical output K/R/t must be supplied together")
    output_camera = None
    if canonical_output_K is not None and canonical_output_R is not None and canonical_output_t is not None:
        _validate_camera(canonical_output_K, canonical_output_R, canonical_output_t, vertices.shape[0])
        output_camera = {"K": canonical_output_K.detach().clone(), "R": canonical_output_R.detach().clone(), "t": canonical_output_t.detach().clone()}
    cache = prepare_dense_mesh_raster_cache(
        vertices, faces, prior_projection_K, prior_projection_R, prior_projection_t,
        image_height=image_size, image_width=image_size, patch_size=patch_size,
        face_region_ids=face_region_ids, vertex_region_ids=vertex_region_ids,
        region_names=region_names, region_binding_label=region_binding_label,
    )
    result = apply_dense_mesh_raster_cache(
        encoded_vertex_features, cache, token_bridge=token_bridge,
        return_dense_features=return_dense_features,
    )
    result["canonical_output_camera"] = output_camera
    return result
