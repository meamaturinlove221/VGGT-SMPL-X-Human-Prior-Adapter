"""冻结选定整份 GW 的固定 V 深度和一次定义的有符号 VN。"""

import argparse
from pathlib import Path

import numpy as np

from research.dna_b_adapter_20261021.paired_train import write_json

from .contract import OUT, V_PATH, read
from .first_surface import mesh_arrays


def first_arrays(render, index=None):
    """读取同一次首面输出，兼容原 QA 的独立三角形索引档。"""
    with np.load(render) as archive:
        first = {key: archive[key] for key in ('K', 'E', 'depth', 'valid')}
        if index is None:
            first['face_id'] = archive['face_id']
            first['mesh'] = archive['mesh'].item()
    if index is not None:
        with np.load(index) as archive:
            first['face_id'] = archive['face_index']
    return first


def face_geometry(mesh, transform=None):
    """返回 mesh 原 winding 的非退化几何面单位法线与面积。"""
    vertices, faces = mesh_arrays(mesh, transform)
    triangle = vertices[faces].astype(np.float64)
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    magnitude = np.linalg.norm(cross, axis=1)
    good = np.isfinite(cross).all(1) & np.isfinite(magnitude) & (magnitude > 1e-12)
    normals = np.full(cross.shape, np.nan, dtype=np.float32)
    normals[good] = cross[good] / magnitude[good, None]
    return vertices, faces, normals, .5 * magnitude


def first_world(first):
    """由原内外参和相机 Z 计算真实世界点与相机出射方向。"""
    y, x = np.indices(first['depth'].shape[-2:], dtype=np.float32)
    k, e = first['K'], first['E']
    y_cam = (y - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
    x_cam = (x - k[:, 0, 2, None, None] - k[:, 0, 1, None, None] * y_cam) / k[:, 0, 0, None, None]
    ray_camera = np.stack((x_cam, y_cam, np.ones_like(x_cam)), axis=-1)
    ray_world = np.einsum('shwi,sij->shwj', ray_camera, e[:, :3, :3])
    center = -np.einsum('si,sij->sj', e[:, :3, 3], e[:, :3, :3])
    points = ray_world * first['depth'][..., None] + center[:, None, None, :]
    return points, ray_world


def first_normals(first, normals):
    """只在实际匹配的有效三角面上提供原有符号法线。"""
    ids = first['face_id']
    hit = first['valid'] & np.isfinite(first['depth']) & (first['depth'] > 0)
    if np.any(ids[hit] < 0) or np.any(ids[hit] >= len(normals)):
        raise ValueError('深度命中缺有效原始三角形索引')
    world = np.full(ids.shape + (3,), np.nan, dtype=np.float32)
    world[hit] = normals[ids[hit]]
    return world, hit


def normal_support(first, normals, hit, rays, v):
    """用正面非掠射和四邻同一连续可见层筛 VN，冲突面不强翻。"""
    # 留下中心像素相机方向明确、四邻均实际命中且不存在深度跳变或反向几何面的支持。
    incidence = -(normals * rays).sum(-1) / np.linalg.norm(rays, axis=-1)
    geometry_ok = np.isfinite(normals).all(-1)
    frontal = incidence > .1
    depth = first['depth']
    footprint = depth / np.minimum(first['K'][:, 0, 0], first['K'][:, 1, 1])[:, None, None]
    middle = (slice(None), slice(1, -1), slice(1, -1))
    neighbors = ((slice(None), slice(None, -2), slice(1, -1)),
                 (slice(None), slice(2, None), slice(1, -1)),
                 (slice(None), slice(1, -1), slice(None, -2)),
                 (slice(None), slice(1, -1), slice(2, None)))
    coherent = np.ones_like(hit[middle])
    for adjacent in neighbors:
        aligned = (normals[middle] * normals[adjacent]).sum(-1) > .5
        contiguous = np.abs(depth[middle] - depth[adjacent]) <= 4 * footprint[middle]
        coherent &= hit[adjacent] & frontal[adjacent] & aligned & contiguous
    boundary_ok = np.zeros_like(hit)
    boundary_ok[middle] = coherent
    # 每个排除像素只归到首个原因，便于复算并保留 V 原样。
    reason = np.zeros_like(hit, dtype=np.uint8)
    reason[v & ~geometry_ok] = 1
    reason[v & geometry_ok & ~frontal] = 2
    reason[v & geometry_ok & frontal & ~boundary_ok] = 3
    vn = v & (reason == 0)
    return vn, reason, incidence


def freeze(mesh, render, index=None, historical_depth=None):
    """缺失固定 V 直接报完整像素和相机计数，禁止裁剪或补零。"""
    folder = OUT / 'teacher_target'
    folder.mkdir(exist_ok=True)
    path = folder / 'RECEIPT.json'
    if path.exists():
        raise FileExistsError(path)
    first = first_arrays(render, index)
    if index is None and Path(first['mesh']).resolve() != mesh.resolve():
        raise ValueError('首面归属 Mesh 不同')
    with np.load(V_PATH) as source:
        v = source['valid'][0]
        camera_ids = source['source_camera_ids'].tolist()
        for key in ('K', 'E'):
            if not np.array_equal(first[key], source[key][0]):
                raise ValueError(f'首面缓存 {key} 与冻结 V 不符')
    hit = first['valid'] & np.isfinite(first['depth']) & (first['depth'] > 0)
    missing = v & ~hit
    record = dict(mesh=str(mesh.resolve()), first_surface=str(render.resolve()),
                  face_index=str(index.resolve()) if index else str(render.resolve()),
                  source_V=str(V_PATH), source_camera_ids=camera_ids, depth_count=int(v.sum()),
                  missing_V=int(missing.sum()), missing_V_per_source=missing.sum((1, 2)).tolist())
    if missing.any():
        record.update(status='FIXED_V_MISSING_REQUIRES_WHOLE_TEACHER_DECISION')
        failure = folder / f'MISSING_V_{mesh.parent.parent.name}.json'
        write_json(failure, record)
        np.savez_compressed(failure.with_suffix('.npz'), missing_bvyx=np.argwhere(missing[None]))
        raise ValueError(f'当前整份教师缺固定 V，详见 {failure}')
    selected_depth = first['depth']
    face_match = np.ones_like(v)
    old_delta = np.zeros_like(first['depth'])
    if historical_depth is not None:
        # 旧 clean 保原 Depth 数值，首面重渲不一致像素只排除法线。
        with np.load(historical_depth) as old:
            selected_depth = old['depth']
            if (not np.array_equal(first['K'], old['K']) or not np.array_equal(first['E'], old['E'])
                    or not bool(old['valid'][v].all()) or not np.isfinite(selected_depth[v]).all()
                    or not bool((selected_depth[v] > 0).all())):
                raise ValueError('旧 clean 固定 V 深度或 K/E 不符')
            old_delta = np.abs(selected_depth - first['depth'])
            face_match = old['valid'] & np.isfinite(old_delta) & (old_delta <= 1e-5)
            record.update(depth_source=str(historical_depth.resolve()),
                          historical_face_match_tolerance_m=1e-5,
                          V_face_match=int((v & face_match).sum()),
                          max_V_reraster_camera_Z_difference_m=float(old_delta[v].max()))
    _, _, face_normals, _ = face_geometry(mesh)
    normals, hit = first_normals(first, face_normals)
    _, rays = first_world(first)
    vn, reason, incidence = normal_support(first, normals, hit & face_match, rays, v)
    reason[v & ~face_match] = 4
    vn &= face_match
    if not vn.any():
        raise ValueError('冻结 VN 为空')
    output = folder / 'TARGETS.npz'
    np.savez_compressed(
        output, depth=np.where(v, selected_depth, np.nan)[None], valid=v[None],
        world_normal=np.where(vn[..., None], normals, np.nan)[None],
        normal_valid=vn[None], normal_reason=reason[None],
        v_bvyx=np.argwhere(v[None]), vn_bvyx=np.argwhere(vn[None]),
        V_face_index=first['face_id'][v], V_signed_incidence=incidence[v],
        V_historical_reraster_abs_camera_Z_m=old_delta[v],
        K=first['K'], E=first['E'])
    reasons = {
        '1_nonfinite_or_degenerate_mesh_face': 1,
        '2_back_facing_or_grazing_signed_face': 2,
        '3_neighbor_visibility_or_orientation_conflict': 3,
        '4_reraster_face_camera_Z_differs_from_original': 4,
    }
    record.update(
        status='FROZEN_GW_TARGETS', target_archive=str(output),
        normal_count=int(vn.sum()), normal_per_source=vn.sum((1, 2)).tolist(),
        normal_exclusions={key: (reason == value).sum((1, 2)).tolist() for key, value in reasons.items()},
        world_sign='Original mesh winding, signed facing incidence > .1; no viewwise flips',
        neighbor_rule='4-neighbor matching hit, frontal, signed normal dot > .5, depth jump <=4 projected pixel footprints',
        absolute_dot=False, V_filtered_by_normal=False, certified_GT=False)
    write_json(path, record)
    print(record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--historical-depth', type=Path)
    args = parser.parse_args()
    freeze(args.mesh, args.render, args.index, args.historical_depth)
