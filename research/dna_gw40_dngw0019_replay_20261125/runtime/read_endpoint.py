"""Read texture geometry and actual optimizer steps from a GW40 candidate endpoint."""

from pathlib import Path
import json
import sys

import numpy as np
from plyfile import PlyData
import torch

from research.dna_gw40_dngw0019_replay_20261125.paths import RESULTS


def read_endpoint(root):
    model = root/'GW_OFFICIAL_DIRECT_DEFAULT'
    checkpoint = model/'chkpnt30000.pth'
    state, step = torch.load(checkpoint, map_location='cpu', weights_only=False)
    optimizer = state[10]
    groups = []
    for group in optimizer['param_groups']:
        params = [optimizer['state'].get(key, {}) for key in group['params']]
        groups.append(dict(name=group['name'], steps=[int(p['step']) for p in params if 'step' in p]))

    artifacts = {}
    for name in ('RAW_UNFILTERED.ply', 'mesh_ours_2pivots.ply', 'mesh_ours_2pivots_post.ply',
                 'mesh_ours_2pivots_post_texture_refined_999.ply'):
        path = model/name
        mesh = PlyData.read(path)
        artifacts[name] = dict(path=str(path), vertices=mesh['vertex'].count, faces=mesh['face'].count,
                               vertex_properties=list(mesh['vertex'].data.dtype.names))
        if name == 'mesh_ours_2pivots_post.ply':
            vertices = np.stack([mesh['vertex'][axis] for axis in 'xyz'], axis=1)
            faces = np.stack(mesh['face']['vertex_indices'])
        if name.endswith('texture_refined_999.ply'):
            textured_xyz = np.stack([mesh['vertex'][axis] for axis in 'xyz'], axis=1)
            textured_faces = np.stack(mesh['face']['vertex_indices'])
            if not np.array_equal(textured_xyz, vertices) or not np.array_equal(textured_faces, faces):
                raise ValueError(f'{path} 的纹理步骤改变了原几何')

    run = json.loads((root/'GW40_RUN.json').read_text(encoding='utf8'))
    log = (root/'GW40_OFFICIAL.log').read_bytes()
    texture_complete = b'1000/1000' in log
    if run['status'] != 'COMPLETED' or step != 30000 or not texture_complete:
        raise ValueError(f'{root} 缺少完成的 30,000 步 GW 检查点或 1,000 步纹理回执')
    receipt = dict(nominal_iteration=step, gaussians=int(state[1].shape[0]), actual_saved_groups=groups,
                   teacher_checkpoint=str(checkpoint), texture_geometry_exact=True, artifacts=artifacts,
                   texture_1000_of_1000_logged=texture_complete, student_optimizer_updates=0, run=run)
    (root/'GW40_ACTUAL_ENDPOINT.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(step, receipt['gaussians'], texture_complete, artifacts, flush=True)
    return receipt


if __name__ == '__main__':
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS/'DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125'
    read_endpoint(base/f'GW40_{sys.argv[1]}')
