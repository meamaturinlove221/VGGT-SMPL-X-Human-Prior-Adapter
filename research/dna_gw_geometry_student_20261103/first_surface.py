"""按原 518 整数像素和原相机提取参考 Mesh 的几何首面。"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from .contract import read


def scan_transform(lineage):
    """读取原 B GIVEN lineage 中来自六源 producer 的真实 scan 变换。"""
    return read(lineage)['source_lineage']['scan_to_model_transform']


def mesh_arrays(path, transform=None):
    """按原面顺序读取完整 mesh，TH 仅应用既有 scan 到 world 转换。"""
    if path.suffix.lower() == '.obj':
        from research.mentor_human_model.data.prepare_train4 import read_obj_geometry

        vertices, faces = read_obj_geometry(path)
        faces = faces.astype(np.int32)
    else:
        mesh = trimesh.load(path, process=False, maintain_order=True)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    if transform is not None:
        matrix = np.asarray(transform['matrix'], dtype=np.float32)
        vertices = float(transform['scale']) * np.einsum('ij,nj->ni', matrix, vertices, optimize=False)
        vertices += np.asarray(transform['translation'], dtype=np.float32)
    return vertices, faces


@torch.no_grad()
def raster(vertices, faces, k, e, shape, context):
    """与原 teacher_quality.render 同投影，增加原几何面索引输出。"""
    import nvdiffrast.torch as dr

    height, width = shape
    k = torch.as_tensor(k, dtype=torch.float32, device='cuda')
    e = torch.as_tensor(e, dtype=torch.float32, device='cuda')
    camera = vertices @ e[:3, :3].T + e[:3, 3]
    x, y, z = camera.unbind(-1)
    near, far = .01, 100.
    clip = torch.stack((
        2 * k[0, 0] / width * x + 2 * k[0, 1] / width * y + (2 * (k[0, 2] + .5) / width - 1) * z,
        2 * k[1, 1] / height * y + (2 * (k[1, 2] + .5) / height - 1) * z,
        (far + near) / (far - near) * z - 2 * far * near / (far - near),
        z,
    ), dim=-1)[None]
    result, _ = dr.rasterize(context, clip, faces, resolution=(height, width))
    depth, _ = dr.interpolate(z.contiguous()[None, :, None], result, faces)
    ids = result[0, :, :, 3].cpu().numpy().astype(np.int32) - 1
    return depth[0, :, :, 0].cpu().numpy(), ids


def render(mesh, inputs, output, lineage=None):
    """只渲染六个原源视角，保存每个像素的真实 hit、camera-Z 和三角形号。"""
    import nvdiffrast.torch as dr

    if output.exists():
        raise FileExistsError(output)
    transform = scan_transform(lineage) if lineage else None
    vertices, faces = mesh_arrays(mesh, transform)
    with np.load(inputs) as inp:
        k, e, shape = inp['K'][0], inp['E'][0], inp['masks'].shape[-2:]
    vertices = torch.as_tensor(vertices, device='cuda')
    faces = torch.as_tensor(faces, device='cuda')
    context = dr.RasterizeCudaContext()
    rows = []
    for view, (kv, ev) in enumerate(zip(k, e)):
        rows.append(raster(vertices, faces, kv, ev, shape, context))
        print(f'Completed original source triangle render {view + 1}/{len(k)}', flush=True)
    depth, ids = (np.stack(values) for values in zip(*rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, K=k, E=e, depth=depth, face_id=ids, valid=ids >= 0,
                        mesh=str(mesh.resolve()), inputs=str(inputs.resolve()),
                        original_source_triangle_renders=len(rows),
                        scan_transform_source=str(lineage.resolve()) if lineage else '')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scan-lineage', type=Path)
    args = parser.parse_args()
    render(args.mesh, args.inputs, args.output, args.scan_lineage)
