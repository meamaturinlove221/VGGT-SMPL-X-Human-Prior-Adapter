"""只用 DN0019 原六源 RAW 在连续像素格增密人体展示点。"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from color_dense_dn0019 import COLORED, read_vertices
from dense_dn0019 import COUNT, MESH, ROOT, SEED

from freeze_support import capsule
from local_0019_cpu_review import panel, project


QA = ROOT.parent / 'DNA_GW_CLEAN_QA_AND_STUDENT_GATE_20261103'
FOUR_MASKS = ROOT.parent / 'DNA_GW_OFFICIAL_DIRECT_20261102/human_body_only/BODY_ONLY_RECEIPT.json'
STEM = 'DN_GW_U200_0019_RAW_INTERPOLATED_BODY_DENSE_RGB_2000000_POINTS'
CAMERAS = [2, 11, 17, 26, 35, 41]
# 原 518 RGB 上可直接辨识的独立地布亮斑/地灯，编号来自审图中的原 8 连通 mask。
FLOOR_COMPONENTS = {2: (3, 4), 11: (2, 7, 8, 14), 17: (2, 7, 13)}
FOOTPRINT_LIMIT = 2.


def read(path):
    """读取原冻结来源，不重估相机或范围。"""
    return json.loads(path.read_text(encoding='utf-8'))


def source_data():
    """原 PLY 和 RAW 网格逐值绑定，RGB 直接使用原彩色 RAW。"""
    folder = MESH.parent
    archive = folder / 'source_RAW.npz'
    ply = folder / 'source_RAW_all_foreground.ply'
    raw = read_vertices(ply, COLORED)
    with np.load(archive) as source:
        keys = ('world_points', 'mask', 'depth', 'K', 'E')
        data = {key: source[key][0] for key in keys}
    fg = data['mask']
    if len(raw) != 91537 or not np.array_equal(raw['xyz'], data['world_points'][fg]):
        raise ValueError('原彩色 PLY 与原 RAW 像素坐标不同')
    index = np.full(fg.shape, -1, dtype=np.int32)
    index[fg] = np.arange(len(raw), dtype=np.int32)
    return raw, data, index, archive, ply


def four_silhouette_masks():
    """复用已冻结四个 TRAIN 原人体 mask、K/E 与原 8px rim。"""
    views = read(FOUR_MASKS)['selection']
    masks = []
    for view in views:
        observed = cv2.imread(view['mask'], cv2.IMREAD_GRAYSCALE)
        rim = cv2.dilate((observed >= 128).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        masks.append((view['camera_id'], np.asarray(view['K']), np.asarray(view['E']), rim))
    return masks


def silhouette_votes(points, masks):
    """在四路已校准原轮廓上计支持，不用三维近邻或 Kinect 残差。"""
    votes = np.zeros(len(points), dtype=np.uint8)
    per_camera = {}
    for camera_id, k, e, mask in masks:
        camera = np.einsum('ij,nj->ni', e[:3, :3], points) + e[:3, 3]
        pixel = np.einsum('ij,nj->ni', k, camera)
        uv = np.rint(pixel[:, :2] / pixel[:, 2:]).astype(int)
        x, y = uv.T
        valid = (camera[:, 2] > 0) & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        support = valid.copy()
        support[valid] = mask[y[valid], x[valid]]
        votes += support
        per_camera[camera_id] = int(support.sum())
    return votes, per_camera


def retain_human(data, raw, index):
    """只去掉已审图的脱离人体地布片，四路支持充分的保留。"""
    masks = four_silhouette_masks()
    keep = data['mask'].copy()
    notes = []
    for view, camera_id in enumerate(CAMERAS):
        labels = cv2.connectedComponentsWithStats(keep[view].astype(np.uint8), connectivity=8)[1]
        for component in FLOOR_COMPONENTS.get(camera_id, ()):
            region = labels == component
            if not region.any():
                raise ValueError(f'审图的原组件不同 camera{camera_id}/{component}')
            votes, per_camera = silhouette_votes(raw['xyz'][index[view][region]], masks)
            y, x = np.where(region)
            remove = not bool((votes >= 3).any())
            if remove:
                keep[view][region] = False
            notes.append(dict(camera=camera_id, original_mask_component=component,
                              pixel_bounds_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                              points=len(votes), train_support_per_camera=per_camera,
                              points_with_at_least_three_votes=int((votes >= 3).sum()),
                              removed=remove, reason='Visually detached studio-floor/light patch'))
    obs = read(QA / 'STUDENT_SINGLE_CASE_2X2_CONTRACT.json')['observation']
    arm = np.asarray(obs['joint_world_xyz'])
    radius = obs['capsule_radius_m']
    before = capsule(raw['xyz'], arm, radius)
    kept_index = index[keep]
    if before[kept_index].sum() != before.sum():
        raise ValueError('环境选择不得删除固定肩肘腕窗口原点')
    return keep, notes, int(before.sum())


def eligible_cells(data, index, keep):
    """只有同源 2×2 四角全有效，且深度跨度与四边均连续才插值。"""
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    shape = keep.shape
    z, world = data['depth'], data['world_points']
    cell_mask = keep & np.isfinite(world).all(-1) & np.isfinite(z) & (z > 0)
    valid = np.logical_and.reduce([cell_mask[:, dy:shape[1]-1+dy, dx:shape[2]-1+dx] for dy, dx in offsets])
    view, y, x = np.where(valid)
    z4 = np.stack([z[view, y + dy, x + dx] for dy, dx in offsets], axis=-1)
    p4 = np.stack([world[view, y + dy, x + dx] for dy, dx in offsets], axis=1)
    f = np.minimum(data['K'][view, 0, 0], data['K'][view, 1, 1])
    limit = FOOTPRINT_LIMIT * z4.min(1) / f
    edges = np.stack([np.linalg.norm(p4[:, a] - p4[:, b], axis=-1)
                      for a, b in ((0, 1), (0, 2), (1, 3), (2, 3))], axis=-1)
    good = (np.ptp(z4, axis=1) <= limit) & (edges.max(1) <= limit)
    cells = np.column_stack((view[good], y[good], x[good])).astype(np.int16)
    vv, yy, xx = cells.T
    parents = np.stack([index[vv, yy + dy, xx + dx] for dy, dx in offsets], axis=1)
    return cells, parents, int(valid.sum())


def fill(result, raw, data, cells, parents, counts, uv):
    """在同格插值 camera-Z/RGB，用原射线反投影世界坐标。"""
    expanded = np.repeat(np.arange(len(cells), dtype=np.int32), counts)
    depth = data['depth']
    offsets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for start in range(0, len(expanded), 200_000):
        stop = min(start + 200_000, len(expanded))
        ids = expanded[start:stop]
        view, y, x = cells[ids].T
        u, v = ((uv[start:stop].astype(np.float64) + .5) / 65536.).T
        weight = np.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), axis=-1)
        z4 = depth[view[:, None], y[:, None] + offsets[:, 0], x[:, None] + offsets[:, 1]]
        z = (z4 * weight).sum(1)
        k, e = data['K'][view], data['E'][view]
        ray_y = (y + v - k[:, 1, 2]) / k[:, 1, 1]
        ray_x = (x + u - k[:, 0, 2] - k[:, 0, 1] * ray_y) / k[:, 0, 0]
        camera = np.stack((ray_x * z, ray_y * z, z), axis=1)
        result['xyz'][start:stop] = np.einsum('ni,nij->nj', camera - e[:, :3, 3], e[:, :3, :3])
        result['rgb'][start:stop] = np.rint(
            np.einsum('ni,nij->nj', weight, raw['rgb'][parents[ids]])).astype(np.uint8)


def save_ply(path, points):
    """输出完整二进制 PLY，回读确认坐标/RGB 对齐且全部有限。"""
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    temporary = path.with_suffix('.partial')
    with temporary.open('xb') as stream:
        stream.write(header.encode('ascii'))
        points.tofile(stream)
    reread = read_vertices(temporary, COLORED)
    if not np.array_equal(reread, points) or not np.isfinite(reread['xyz']).all():
        raise ValueError('完整输出读回不同或有非有限点')
    temporary.replace(path)


def preview(path, original, result):
    """同一世界窗口查看原 RAW 和全部密集点，保留手臂和鞋的轮廓。"""
    angles = (0, 90, 180)
    w, h = 390, 440
    image = Image.new('RGB', (w * len(angles), h * 2), 'white')
    for col, angle in enumerate(angles):
        coords = project(original['xyz'], angle)[:, :2]
        low, high = coords.min(0) - .06, coords.max(0) + .06
        box = (low[0], high[0], low[1], high[1])
        for row, (name, points) in enumerate((('ORIGINAL FOREGROUND RAW', original),
                                               ('DENSE SAME-SOURCE RAW INTERPOLATION', result))):
            tile = panel(points['xyz'], points['rgb'], None, angle, box, (w, h))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, w, 21), fill='white')
            draw.text((4, 4), f'{name} | {angle} deg', fill='black')
            image.paste(tile, (col * w, row * h))
    image.save(path)


def main():
    """完成点云、可重算的父像素和原点保留索引。"""
    folder = ROOT / 'display_dense/RAW_BODY_DENSE_RGB'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'{STEM}.ply'
    if output.exists():
        raise FileExistsError(output)
    raw, data, index, archive, source = source_data()
    keep, cleaning, arm_original = retain_human(data, raw, index)
    cells, parents, before_continuity = eligible_cells(data, index, keep)
    kept = index[keep]
    n_new = COUNT - len(kept)
    rng = np.random.default_rng(SEED)
    counts = np.full(len(cells), n_new // len(cells), dtype=np.int32)
    counts[rng.permutation(len(cells))[:n_new % len(cells)]] += 1
    uv = rng.integers(0, 65536, size=(n_new, 2), dtype=np.uint16)
    result = np.empty(COUNT, dtype=COLORED)
    result[:len(kept)] = raw[kept]
    fill(result[len(kept):], raw, data, cells, parents, counts, uv)
    save_ply(output, result)
    provenance = folder / 'POINT_PROVENANCE.npz'
    removed = index[data['mask'] & ~keep]
    np.savez_compressed(provenance, camera_ids=np.asarray(CAMERAS), original_raw_index=kept,
                        original_view_y_x=np.argwhere(keep).astype(np.int16), removed_original_raw_index=removed,
                        removed_view_y_x=np.argwhere(data['mask'] & ~keep).astype(np.int16),
                        cell_source_y_x=cells, cell_parent_raw_indices=parents, cell_generated_counts=counts,
                        generated_uv_q16=uv)
    figure = folder / 'RAW_BODY_DENSE_RGB_PREVIEW.png'
    preview(figure, raw, result)
    report = dict(status='RAW_BODY_DENSE_RGB_COMPLETE', output=str(output), output_bytes=output.stat().st_size,
                  source_RAW_PLY=str(source), source_RAW_archive=str(archive), seed=SEED,
                  original_RAW_points=len(raw), removed_original=len(removed), retained_original=len(kept),
                  new_interpolated=n_new, points=COUNT, original_foreground_per_source=data['mask'].sum((1, 2)).tolist(),
                  retained_original_per_source=keep.sum((1, 2)).tolist(),
                  generated_per_source=np.bincount(cells[:, 0], weights=counts, minlength=6).astype(int).tolist(),
                  original_arm_window_points_before=arm_original, original_arm_window_points_retained=arm_original,
                  original_points_and_RGB_exact=bool(np.array_equal(result[:len(kept)], raw[kept])),
                  cleaning=dict(audit=str(folder / 'ORIGINAL_SOURCE_MASK_ENVIRONMENT_AUDIT.png'),
                                masks=str(FOUR_MASKS), components=cleaning,
                                rule='Only visually identified detached floor patches, and no point in the component supported by at least three of four fixed TRAIN masks with 8px rim. Uncertain or attached pixels retained.'),
                  camera17='MASK_SEMANTICS_UNRESOLVED. Original source mask remains unchanged. Display-copy-only floor removal.',
                  interpolation=dict(candidate_cells=before_continuity, accepted_cells=len(cells),
                                     rejected_discontinuous_cells=before_continuity - len(cells),
                                     rule='All four adjacent pixels are retained finite positive-depth points from the same source. Camera-Z span and all four world edges <= 2 * minimum corner Z / min(fx,fy). Bilinear camera-Z and RAW RGB, then calibrated subpixel ray unprojection.',
                                     threshold_is_engineering_display_only=True),
                  provenance=str(provenance), provenance_rule='First retained_original rows follow original_raw_index. Remaining rows group by cell order and cell_generated_counts; each uv=(generated_uv_q16+.5)/65536, with parents ordered TL/TR/BL/BR.',
                  preview=str(figure), readback_xyz_and_rgb_exact=True,
                  scope='RAW-derived display interpolation. Interpolated points are not new network predictions or GT. Unknown mask pixels and residual cross-view geometry errors may remain.',
                  network=0, mesh=0, DPSR=0, refiner=0, GPU=0, optimizer=0)
    (folder / 'RECEIPT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
