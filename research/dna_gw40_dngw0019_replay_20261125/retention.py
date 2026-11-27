"""对照同输入的历史 0019 DN_GW 和同帧学生的固定 18 例。"""
import csv
import json
from pathlib import Path
import sys

from replay import OUT, RESULTS

HIST = RESULTS/'DNA_GW_GEOMETRY_STUDENT_20261103/evaluation'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def compare(code):
    farm = OUT/f'GW40_{code}'/'matched_student'
    index = farm/'evaluation/DN_GW_U200/EVALUATION_RECEIPT.json'
    old = {r['id']:r for r in read(HIST/'DN_GW_U200/EVALUATION_RECEIPT.json')}
    new = {r['id']:r for r in read(index)}
    manifest = read(HIST/'ENDPOINT18_MANIFEST.json')
    contract = read(farm/'TRAINING_CONTRACT.json')
    target = contract['sample']
    values = []
    for sample in manifest['samples']:
        sid = sample['id']
        a,b = old[sid],new[sid]
        assert all(a[k]==b[k] for k in ('input_npz','frozen_Xf','prior_role','camera_role'))
        assert a['metrics']['valid_pixels']==b['metrics']['valid_pixels']
        row = dict(sample=sid,role=manifest['roles'][sid],valid_pixels=a['metrics']['valid_pixels'])
        if sid == target:
            row['matched_exposure'] = 'SUPERVISED_TRAINING_FRAME'
        elif sid in contract['exposures_per_branch']:
            row['matched_exposure'] = 'COMPANION_TRAINING_FRAME'
        elif sid.split('/')[:2] == target.split('/')[:2]:
            row['matched_exposure'] = 'SAME_IDENTITY_OTHER_FRAME'
        else:
            row['matched_exposure'] = 'OUTSIDE_THIS_UPDATE_DRAW'
        for k in ('depth_MAE','depth_P95','Direct_mean','Direct_P95'):
            before, after = a['metrics'][k], b['metrics'][k]
            row.update({f'original_{k}':before,f'matched_{k}':after,f'delta_{k}':after-before})
        row.update(original=str(Path(a['output'])/'INFERENCE.json'),matched=str(Path(b['output'])/'INFERENCE.json'))
        values.append(row)
    output = farm/'FIXED18_VS_ORIGINAL_DN.csv'
    with output.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=values[0])
        writer.writeheader()
        writer.writerows(values)
    for role in ('FIXED_DNA8','FIXED_TH8','TRAIN_HARD2'):
        rows=[row for row in values if row['role']==role]
        if role == 'FIXED_DNA8' and code == '0034':
            rows = [row for row in rows if row['matched_exposure'] != 'SUPERVISED_TRAINING_FRAME']
        for key in ('depth_MAE','Direct_mean'):
            original=sum(r[f'original_{key}'] for r in rows)/len(rows)
            matched=sum(r[f'matched_{key}'] for r in rows)/len(rows)
            print(code,role,len(rows),key,original,matched,matched-original)
    print(output)


if __name__=='__main__':
    compare(sys.argv[1])
