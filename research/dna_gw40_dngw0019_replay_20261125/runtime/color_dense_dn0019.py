"""把同一 DN0019 的原 RAW RGB 迁移到既有 200 万个展示点。"""

import json

import numpy as np
from scipy.spatial import cKDTree

from dense_dn0019 import COUNT, MESH, ROOT, STEM


COLORED = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])


def read_vertices(path, dtype):
    """读取绑定的二进制小端 PLY 顶点，不转换原世界坐标。"""
    with path.open('rb') as stream:
        header = []
        for line in stream:
            if line.strip() == b'end_header':
                break
            header.append(line.strip())
        if header[:2] != [b'ply', b'format binary_little_endian 1.0']:
            raise ValueError(f'PLY 格式不符 {path}')
        count = int(next(line for line in header if line.startswith(b'element vertex ')).split()[-1])
        result = np.fromfile(stream, dtype=dtype, count=count)
        if len(result) != count:
            raise ValueError(f'PLY 实际顶点不足 {path}')
        return result


def transfer():
    """原 XYZ 保持逐值相同，使用最近原 RAW 的字节颜色。"""
    directory = ROOT / 'display_dense'
    original = directory / f'{STEM}.ply'
    source = MESH.with_name('source_RAW_all_foreground.ply')
    output = directory / f'{STEM}_RGB.ply'
    receipt = directory / f'{STEM}_RGB_RECEIPT.json'
    if output.exists() or receipt.exists():
        raise FileExistsError(output)
    points = read_vertices(original, np.dtype(('<f4', (3,))))
    raw = read_vertices(source, COLORED)
    if len(points) != COUNT or len(raw) != 91537:
        raise ValueError('原 200 万坐标或同 DN 的 91,537 RAW 点数量不同')
    tree = cKDTree(raw['xyz'])
    distances, indices = tree.query(points, k=1, workers=4)
    colors = raw['rgb'][indices]
    combined = np.empty(COUNT, dtype=COLORED)
    combined['xyz'] = points
    combined['rgb'] = colors
    header = (
        f'ply\nformat binary_little_endian 1.0\nelement vertex {COUNT}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    ).encode('ascii')
    temp = output.with_suffix('.partial')
    with temp.open('xb') as stream:
        stream.write(header)
        combined.tofile(stream)
    verified = read_vertices(temp, COLORED)
    if (temp.stat().st_size != len(header) + COUNT * COLORED.itemsize
            or not np.array_equal(verified['xyz'].view('<u4'), points.view('<u4'))
            or not np.array_equal(verified['rgb'], colors)):
        raise ValueError('RGB PLY 实际坐标、颜色或长度不符')
    temp.replace(output)
    d_mm = distances * 1000
    record = dict(
        status='DISPLAY_RAW_COLOR_TRANSFER_COMPLETE',
        source_final_mesh=str(MESH), source_final_mesh_color='Original mesh has no RGB',
        input_sampled_xyz_ply=str(original), source_RAW_color_ply=str(source), source_RAW_points=len(raw),
        method='Nearest neighbor in original shared metric XYZ; copy original raw uint8 RGB',
        count=COUNT, xyz_exact_float32_bits=True, rgb_dtype='uchar',
        unique_RAW_color_donors=int(np.unique(indices).size),
        nearest_RAW_distance_mm=dict(
            min=float(d_mm.min()), mean=float(d_mm.mean()), median=float(np.median(d_mm)),
            p95=float(np.percentile(d_mm, 95)), p99=float(np.percentile(d_mm, 99)), max=float(d_mm.max())),
        points_over_25mm=int((d_mm > 25).sum()), points_over_50mm=int((d_mm > 50).sum()),
        scope='Display color transfer from original DN source RGB. Not predicted novel colors, texture truth, or new RAW.',
        limitation='No visibility filtering: transfer can borrow colors from another view or a nearby layer, particularly where distances are large.',
        alignment=0, geometry_changes=0, network=0, training=0, render=0, GPU=0,
        output_ply=str(output), output_bytes=output.stat().st_size,
    )
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    transfer()
