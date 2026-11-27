"""只准备三个已恢复身份的48真实相机；复用cam00，不运行任何模型或重建。"""
from pathlib import Path
import csv
import importlib.util
import json
import math
import shutil
import traceback

import cv2
import h5py
import numpy as np

from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, REPO as ROOT

OUT = ARTIFACTS/'DNA_MULTISUBJECT_REPAIR_20261024/inputs'
SELECTED = ['0047_01', '0094_02', '0102_02']


def read(path):
    """读取已冻结的真实帧和恢复回执。"""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    """逐相机保存一次读取事实与失败，不改旧回执。"""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def freeze_split(row, records):
    """只以标定中心做既有FPS，真实学生六源和cam00保护在训练组。"""
    centers = {r['camera_id']: np.asarray(r['official_RT_c2w'])[:3, 3] for r in records}
    protected = sorted({0, *map(int, row['source_view_ids'])})
    eligible = sorted(set(centers) - set(protected))
    held = [min(eligible)] if eligible else []
    while len(held) < min(8, len(eligible)):
        remaining = [cid for cid in eligible if cid not in held]
        held.append(max(remaining, key=lambda cid: (min(float(np.sum((centers[cid]-centers[j])**2)) for j in held), -cid)))
    train = sorted(set(centers) - set(held))
    return dict(sample_id=row['sample_id'], train=train, heldout_sorted=sorted(held), heldout_FPS_order=held,
                protected_train=protected, source_view_ids=row['source_view_ids'], missing_protected_train=sorted(set(protected)-set(centers)),
                train_count=len(train), heldout_count=len(held),
                rule='Official c2w camera centers only; eligible excludes original six-source and cam00; seed smallest eligible ID; maximin Euclidean FPS, ties smallest physical ID; no RGB/mask/model errors used.',
                source_only_prior='NOT_CONSUMED_THIS_PREPARATION; no GIVEN substitution',
                heldout_role='Validation only; excluded from fitting, carving, mask edits, bbox and teacher optimization',
                bbox_created=False)


def show_and_index(folder, records, split):
    """画小联系图并记录完整物理ID和中心角度，不推断人体朝向。"""
    center_mean = np.mean([np.asarray(r['official_RT_c2w'])[:3,3] for r in records], axis=0)
    by_id = {r['camera_id']:r for r in records}
    canvas = np.full((6*156, 8*153, 3), 240, np.uint8)
    table = []
    for cid in range(48):
        record = by_id.get(cid)
        group = 'TRAIN' if cid in split['train'] else 'HELD' if cid in split['heldout_sorted'] else 'UNREADABLE'
        entry = dict(camera_id=cid, camera_name=f'{cid:02d}', status='READ_SUCCESS' if record else 'UNREADABLE', split=group,
                     student_source=cid in set(map(int, split['source_view_ids'])), center_x='', center_y='', center_z='',
                     azimuth_xz_degrees='', elevation_y_degrees='', radius_from_camera_center_mean='')
        x,y = (cid%8)*153, (cid//8)*156
        if record:
            center = np.asarray(record['official_RT_c2w'])[:3,3]; vector = center-center_mean
            entry.update(center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
                         azimuth_xz_degrees=math.degrees(math.atan2(vector[2],vector[0])),
                         elevation_y_degrees=math.degrees(math.atan2(vector[1],math.hypot(vector[0],vector[2]))),
                         radius_from_camera_center_mean=float(np.linalg.norm(vector)))
            image = cv2.imread(record['rgb'])
            canvas[y:y+128, x:x+153] = cv2.resize(image,(153,128),interpolation=cv2.INTER_AREA)
            del image
        label = f'{cid:02d} {group}' + (' SRC' if entry['student_source'] else '')
        cv2.putText(canvas,label,(x+3,y+146),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,180) if group=='HELD' else (0,0,0),1,cv2.LINE_AA)
        table.append(entry)
    cv2.imwrite(str(folder/'CONTACT48.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    with (folder/'CAMERA48_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(folder/'CAMERA_ANGLE_CONVENTION.json',dict(center_reference='Mean of successful official calibrated camera centers; not estimated body position',
          center_reference_xyz=center_mean.tolist(),azimuth='atan2(center_z-reference_z, center_x-reference_x), degrees',
          elevation='atan2(center_y-reference_y, hypot(center_x-reference_x,center_z-reference_z)), degrees',
          interpretation='Coordinate-space camera layout only; no body-facing or semantic pose claim'))


def main():
    """每帧只读剩余47相机，保存小尺寸prepared和冻结分组。"""
    spec=importlib.util.spec_from_file_location('teacher_data',ROOT/'research/dna_b_adapter_20261021/teacher_data.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    manifest=read(OUT/'TRAIN_SOURCE_MANIFEST.json')['train_frames']
    recovered={r['sample_id']:r for r in read(OUT/'DOWNLOAD_RECOVERY_CAM00_RECEIPT.json')['rows']}
    summary=[]
    for sequence in SELECTED:
        row=next(r for r in manifest if r['sequence']==sequence and r['frame']==0)
        source=recovered[row['sample_id']];folder=OUT/row['sample_id'].replace('/','__')
        receipt_path=folder/'RECOVERED_READ48_RECEIPT.json'
        if receipt_path.exists():
            raise FileExistsError(f'Already attempted {sequence}; inspect receipt instead of rereading')
        free_before=shutil.disk_usage(OUT).free
        if free_before<512*1024**2:
            raise OSError('Insufficient disk for remaining bounded prepared outputs')
        records=[read(folder/'CAM00_RECORD.json')]
        receipt=dict(sample_id=row['sample_id'],main_archive=source['main_archive'],annotation_archive=source['annotation_archive'],
                     source_view_ids=row['source_view_ids'],cam00_reused=True,cam00_record=str(folder/'CAM00_RECORD.json'),
                     free_bytes_before=free_before,estimated_three_frames_prepared_bytes=120*1024**2,
                     records=records,failures=[],new_camera_attempts=0,new_camera_decodes=0,status='READING_REMAINING47',
                     new_network_forwards=0,new_GPU_calls=0,new_pose_calls=0,new_VH_calls=0,new_NCC_calls=0,bbox_created=False)
        write(receipt_path,receipt)
        with h5py.File(source['main_archive'],'r') as main,h5py.File(source['annotation_archive'],'r') as annots:
            for cid in range(1,48):
                receipt['new_camera_attempts']+=1
                try:
                    record,mask=reader.export_view(main,annots,cid,0,folder,cid)
                    records.append(record);receipt['new_camera_decodes']+=1
                    del mask
                except Exception as exc:
                    receipt['failures'].append(dict(camera_id=cid,rgb_key=f'Camera_5mp/{cid}/color/0',mask_key=f'Mask/{cid}/mask/0',
                                                   error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
                write(receipt_path,receipt)
                print(sequence,'cam',cid,'success',len(records),'fail',len(receipt['failures']),flush=True)
        split=freeze_split(row,records)
        write(folder/'SPLIT40_8.json',split)
        reader.nhr_cameras(folder,records)
        show_and_index(folder,records,split)
        receipt.update(status='COMPLETE48' if len(records)==48 else 'INCOMPLETE48_NO_RETRY',split=str(folder/'SPLIT40_8.json'),
                       camera_index=str(folder/'CAMERA48_INDEX.csv'),contact_sheet=str(folder/'CONTACT48.jpg'),
                       free_bytes_after=shutil.disk_usage(OUT).free)
        write(receipt_path,receipt)
        summary.append(dict(sample_id=row['sample_id'],status=receipt['status'],successful_cameras=len(records),
                            new_camera_attempts=receipt['new_camera_attempts'],new_camera_decodes=receipt['new_camera_decodes'],
                            train=split['train_count'],heldout=split['heldout_count'],receipt=str(receipt_path)))
        write(OUT/'RECOVERED_THREE48_SUMMARY.json',dict(rows=summary,completed=len(summary)==3,new_model_calls=0,new_VH_calls=0,new_NCC_calls=0))
        print('FINISHED',sequence,receipt['status'],flush=True)


if __name__=='__main__':main()
