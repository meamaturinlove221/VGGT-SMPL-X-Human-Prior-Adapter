"""根据已登记样本的原物理帧生成 GW 与独立 QA 输入。"""
from pathlib import Path

import numpy as np
from PIL import Image

from replay import BODY, NEW_FRAMES, PREVIOUS, SOURCE, load, read48, row_for, sample, save


def student_camera_errors(row, records):
    with np.load(row['input_npz']) as inp:
        extrinsics = inp['E'][0]
    delta = np.abs(extrinsics[:, None] - np.array([r['E'] for r in records])[None]).max((2,3))
    closest = np.sort(delta, axis=1)
    if (delta.argmin(1).tolist() != SOURCE or np.any(closest[:,0] > 1e-5)
            or np.any(closest[:,1] < 1e-3)):
        raise ValueError('学生实际外参与原物理相机不一致')
    return closest[:,0].tolist()


def teacher_input(code, run):
    split, source_path, train, held, records = read48(code)
    original = load(source_path)
    rows = []
    for record in records:
        width, height = record['output_hw'][::-1]
        with Image.open(record['rgb']) as rgb, Image.open(record['mask']) as mask:
            if rgb.size != mask.size or rgb.size != (width, height) or (width,height) != (1224,1024):
                raise ValueError(f'RGB/mask 与标定画幅不符 {record["camera_id"]}')
            foreground = int(np.count_nonzero(np.asarray(mask.convert('L')) >= 128))
        cid = record['camera_id']
        rows.append(dict(record,subset='train' if cid in train else 'held',
                         rgb_header=dict(wh=[width,height]),mask_pixels=foreground))
    result = dict(sample=sample(code), train_ids=train, held_ids=held, rows=rows,
                  role_receipt=str(split), source_receipt=str(source_path),
                  source_main=original['main_archive'], source_annotation=original['annotation_archive'],
                  four_body_mask_pixels={str(cam):rows[cam]['mask_pixels'] for cam in BODY},
                  mask_handling='Same-frame acquired foreground mask. No 0019 cam09 manual polygon transferred.')
    if code in NEW_FRAMES:
        if any(not rows[cid]['mask_pixels'] for cid in SOURCE + held):
            raise ValueError(f'{sample(code)} 的原六源或 HELD 前景为空')
        result.update(student_camera_ids=SOURCE, student_max_E_delta=student_camera_errors(row_for(code), rows))
    save(run/'FRAME_INPUT.json', result)
    return result


def qa_input(code, qa):
    """沿用共同 3D 观察窗定义，依据本帧 SMPL-X 顶点生成各部位的框。"""
    split, receipt, train, held, records = read48(code)
    row = row_for(code)
    original = load(PREVIOUS/'GW40_TEACHER_QA/0008/FROZEN_TEACHER_QA.json')
    shared = ('historical_train','historical_held','historical_input','historical_body_receipt',
              'body_filter','held_definition','kinect_definition','historical_held_scores',
              'historical_native_scores','historical_native_scope','crops_scope')
    frozen = {key:original[key] for key in shared}
    with np.load(row['input_npz']) as inp, np.load(qa.SMPL) as model:
        points = inp['smpl_vertices'][0]
        faces = inp['smpl_faces'].reshape(-1,3)
        if not np.array_equal(faces, model['f']):
            raise ValueError('SMPL-X 拓扑与观察窗约定不同')
        band = float(inp['unit_scale'].item()) / 128
        parts = model['weights'].argmax(1)
    bounds = {'SUBJECT':np.stack([points.min(0) - .10,points.max(0) + .10]).tolist()}
    for name, joints in qa.GROUPS.items():
        section = points[np.isin(parts, joints)]
        bounds[name] = np.stack([section.min(0) - .06,section.max(0) + .06]).tolist()
    rows = {r['camera_id']:r for r in records}
    for cam in BODY+held:
        with Image.open(rows[cam]['mask']) as image:
            if image.size != (1224,1024) or not np.any(np.asarray(image.convert('L')) >= 128):
                raise ValueError(f'原投影支持缺失 {cam}')
    depth = load(row['lineage'])['source_lineage']['depth_source'].split('::',1)[0]
    if code in NEW_FRAMES and not Path(depth).is_file():
        raise FileNotFoundError(depth)
    frozen.update(sample=row['id'], split_receipt=str(split), prepared_receipt=str(receipt),
                  train=train, held=held, six_student_sources=SOURCE, body_train=BODY,
                  kinect_smc=depth, region_boxes_world_m=bounds, kinect_band_m=band,
                  original_inputs=row['input_npz'], original_targets=row['target_npz'],
                  crop_source=f"{Path(row['input_npz'])} SMPL-X world vertices with {qa.SMPL} dominant weights")
    return frozen, rows
