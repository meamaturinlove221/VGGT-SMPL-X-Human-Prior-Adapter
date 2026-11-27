"""从已保存的 DN0019 全部 Final 三角面生成定种子的面积采样显示点。"""

import json
from pathlib import Path

import numpy as np
import trimesh


from research.dna_gw40_dngw0019_replay_20261125.paths import RESULTS

ROOT = RESULTS/'DNA_GW_GEOMETRY_STUDENT_20261103'
MESH = (ROOT / 'evaluation/DN_GW_U200/DNA_RENDERING__0019__0019_06__0224'
        / 'GIVEN_PRIOR/clothed_mesh.ply')
COUNT = 2_000_000
SEED = 20261103
STEM = 'DN_GW_U200_0019_FINAL_MESH_SAMPLED_2000000_POINTS'


def sample():
    """保留实际每个组件，按三角面积抽面并在面内均匀取点。"""
    dest = ROOT / 'display_dense'
    dest.mkdir(exist_ok=True)
    output, receipt = dest / f'{STEM}.ply', dest / f'{STEM}_RECEIPT.json'
    if output.exists() or receipt.exists():
        raise FileExistsError(output)
    mesh = trimesh.load(MESH, process=False, maintain_order=True)
    if mesh.visual.kind is not None:
        raise ValueError('此轮绑定的是实际没有颜色属性的 DN0019 Final 网格')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    areas = mesh.area_faces
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1)
    components = sorted(components, key=len, reverse=True)
    labels = np.empty(len(faces), dtype=np.int32)
    for component, members in enumerate(components):
        labels[members] = component
    rng = np.random.default_rng(SEED)
    choice = rng.choice(len(faces), size=COUNT, p=areas / areas.sum())
    sampled_components = np.bincount(labels[choice], minlength=len(components))
    if not sampled_components.all():
        raise ValueError('固定采样未覆盖全部输入组件')
    header = (
        f'ply\nformat binary_little_endian 1.0\nelement vertex {COUNT}\n'
        'property float x\nproperty float y\nproperty float z\nend_header\n'
    ).encode('ascii')
    temp = output.with_suffix('.partial')
    with temp.open('xb') as stream:
        stream.write(header)
        for block in np.array_split(choice, 8):
            u, v = rng.random((2, len(block)))
            s = np.sqrt(u)
            weights = np.stack((1 - s, s * (1 - v), s * v), axis=-1)
            triangles = vertices[faces[block]]
            points = np.einsum('ni,nij->nj', weights, triangles).astype('<f4')
            if not np.isfinite(points).all():
                raise ValueError('生成了非有限坐标')
            stream.write(points.tobytes())
    if temp.stat().st_size != len(header) + COUNT * 12:
        raise ValueError('输出 PLY 数据长度与实际点数不同')
    temp.replace(output)
    record = dict(
        status='DISPLAY_SURFACE_SAMPLING_COMPLETE',
        source_mesh=str(MESH), source_vertices=len(vertices), source_faces=len(faces),
        source_components=len(components), source_degenerate_faces=int((areas == 0).sum()),
        components=[dict(faces=len(members), surface_area_m2=float(areas[members].sum()),
                         emitted_points=int(sampled_components[index]))
                    for index, members in enumerate(components)],
        requested_points=COUNT, emitted_points=int(sampled_components.sum()), seed=SEED,
        method='Triangle-area weighted selection and uniform barycentric sampling in original world coordinates',
        color='XYZ only. Input mesh has no RGB properties.',
        scope='Display samples on the existing student Final mesh, not RAW predictions or ground truth.',
        smoothing=0, cleanup=0, hole_filling=0, network=0, DPSR=0, refiner=0, optimizer=0,
        output_ply=str(output), output_bytes=output.stat().st_size,
    )
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    sample()
