"""核对新登记帧的实际六源，并将独立诊断训练授权绑定到冻结的教师首面。"""

from pathlib import Path
import sys

from replay import OUT, SOURCE, load, row_for, sample, save, student_farm


CASES = ('0097', '0121', '0034')


def read_student_source(code):
    import h5py
    import numpy as np
    import torch

    row = row_for(code)
    lineage = load(row['lineage'])
    native = lineage['source_lineage']
    annotation = Path(native['calibration_source'].split('::')[0])
    actor, sequence, frame = row['id'].split('/')[1:]
    if (lineage['sample_id'] != row['id'] or native['native_actor_or_subject'] != actor
            or native['native_sequence_or_pose'] != sequence or native['native_frame'] != int(frame)
            or list(map(int, native['native_rgb_camera_ids'])) != SOURCE
            or lineage['snapshots'] != row['snapshots']):
        raise ValueError(f"{row['id']} 原六源样本回执不匹配")
    packet = torch.load(row['snapshots'], map_location='cpu', mmap=True, weights_only=False)
    aggregator = torch.load(packet['actual_aggregator_input'], map_location='cpu', mmap=True, weights_only=False)
    cameras = []
    with (np.load(row['input_npz']) as inputs, np.load(row['source_path']) as source,
          h5py.File(annotation, 'r') as calibration):
        if (packet['sample_id'] != row['id'] or not torch.equal(packet['RGB'], aggregator)
                or not np.array_equal(packet['RGB'].numpy(), inputs['images'])
                or not np.array_equal(source['rgb'], inputs['images'][0])
                or not np.array_equal(source['K'], inputs['K'][0])
                or not np.array_equal(source['R'], inputs['E'][0, :, :3, :3])
                or not np.array_equal(source['t'], inputs['E'][0, :, :3, 3])):
            raise ValueError(f"{row['id']} 聚合器输入、缓存相机或 RGB 与冻结输入不同")
        for slot, cam in enumerate(SOURCE):
            rgb = native['rgb_paths'][slot]
            mask = native['mask_paths'][slot]
            if (rgb.split('::')[-1] != f'Camera_5mp/{cam}/color/{int(frame)}'
                    or mask.split('::')[-1] != f'Mask/{cam}/mask/{int(frame)}'):
                raise ValueError(f'原始图像与物理相机 {cam} 不同')
            group = calibration[f'Camera_Parameter/{cam:02d}']
            k = group['K'][...].astype(float)
            k[0] *= 433 / 2048
            k[1] *= 518 / 2448
            k[0, 2] += lineage['preprocess']['left_by_view'][slot]
            e = np.linalg.inv(group['RT'][...])
            k_delta = float(abs(k - inputs['K'][0, slot]).max())
            e_delta = float(abs(e - inputs['E'][0, slot]).max())
            if k_delta >= 1e-3 or e_delta >= 1e-6:
                raise ValueError(f'物理相机 {cam} 标定没有对上实际网络六源')
            cameras.append(dict(slot=slot, physical_camera=cam, rgb=rgb, mask=mask,
                                K_max_delta_px=k_delta, E_max_delta=e_delta))
    path = OUT/f'GW40_{code}'/'STUDENT_SOURCE_READBACK.json'
    save(path, dict(sample=row['id'], input=row['input_npz'], target=row['target_npz'],
                    source=row['source_path'], frozen_Xf=row['snapshots'], calibration=str(annotation),
                    aggregator_input=packet['actual_aggregator_input'], RGB_arrays_identical=True,
                    snapshot_layers={str(i):list(t.shape) for i,t in packet['snapshots'].items()},
                    cameras=cameras))
    return path


def authorize_run(code, base):
    if code not in CASES or base != OUT:
        raise ValueError('独立诊断授权只适用于本轮三个指定帧')
    run = base/f'GW40_{code}'
    farm = student_farm(code)
    qa = base/'GW40_TEACHER_QA'/code
    summary_file = qa/'TEACHER_QA_SUMMARY.json'
    summary = load(summary_file)
    verdict_file = qa/'TEACHER_ADMISSION.json'
    verdict = load(verdict_file) if verdict_file.is_file() else summary
    proof = load(qa/'ORIGINAL_SUBSET_READBACK.json')
    source_file = run/'STUDENT_SOURCE_READBACK.json'
    source = load(source_file)
    bound = load(run/'FRAME_INPUT.json')
    target_file = farm/'SERIALIZED_TARGET_READBACK.json'
    target = load(target_file)
    mesh = run/'human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply'
    row = row_for(code)
    if (any(entry['sample'] != row['id'] for entry in (source, bound, summary, verdict, target))
            or source['input'] != target['original_input'] or source['target'] != target['original_target']
            or source['frozen_Xf'] != row['snapshots'] or not source['RGB_arrays_identical']
            or any(Path(path) != mesh for path in (target['mesh'], summary['meshes']['BODY'], verdict.get('mesh',mesh)))
            or not (proof['triangle_original_mapping_equal'] and proof['point_world_xyz_rgb_equal'])
            or not mesh.is_file() or min(sum(target['V']),sum(target['VN'])) <= 0):
        raise ValueError('本帧诊断的实际输入、BODY 或冻结双目标不匹配')
    path = qa/'DIAGNOSTIC_AUTHORIZATION.json'
    save(path, dict(sample=sample(code), mesh=str(mesh), status='AUTHORIZED_ISOLATED_DIAGNOSTIC',
                    authorized_for_isolated_diagnostic=True,
                    authorization='HISTORICAL_MATCHED_FRAME_DIAGNOSTIC',
                    source_qa_status=verdict['status'], evaluation_summary=str(summary_file),
                    independent_decision=str(verdict_file) if verdict_file.is_file() else None,
                    original_subset=str(qa/'ORIGINAL_SUBSET_READBACK.json'), split=bound['role_receipt'],
                    source_readback=str(source_file), target_readback=str(target_file),
                    same_frame_exposure='IN_SAMPLE', optimizer_update_budget=200))
    return path


def record_visual_review(code, status, admitted, images, observations):
    qa = OUT/'GW40_TEACHER_QA'/code
    summary_file = qa/'TEACHER_QA_SUMMARY.json'
    summary = load(summary_file)
    if summary['sample'] != sample(code) or not all((qa/image).is_file() for image in images):
        raise ValueError('审图记录与本例独立 QA 不同')
    save(qa/'TEACHER_ADMISSION.json', dict(
        sample=sample(code), mesh=summary['meshes']['BODY'], status=status,
        admitted_for_matched_reproduction=admitted, evaluation_summary=str(summary_file),
        original_subset_readback=str(qa/'ORIGINAL_SUBSET_READBACK.json'),
        body_held8_micro_iou=summary['HELD8']['BODY']['IoU'],
        native_reliable=summary['native_Kinect8_reliable']['BODY'],
        reviewed=images, observations=observations))
    authorize_run(code, OUT)


if __name__ == '__main__':
    for code in sys.argv[1:]:
        print(read_student_source(code))
