from pathlib import Path
import json
import sys
import h5py
import numpy as np

from research.dna_b_adapter_20261021 import teacher_data as reader
from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, DATASETS, OUTPUT

out = OUTPUT
data = DATASETS/'dna_rendering'
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
code=sys.argv[1]; seq,frame={'0097':('0097_04',0),'0121':('0121_02',135),'0034':('0034_04',0)}[code]
sid=f'DNA_RENDERING/{code}/{seq}/{frame:04d}'; folder=out/f'GW40_{code}'/'role_inputs';folder.mkdir(parents=True,exist_ok=True)
main=data/('fixed_source_recovery_20261016/main' if code=='0034' else 'dna_rendering_part1_main')/(seq+'.smc')
ann=data/'dna_rendering_part1_annotations'/(seq+'_annots.smc')
if code=='0097':
    old=ARTIFACTS/'DNA_TEACHER_COMPARE_20261023/teacher/0097_04_0000/INPUT_RECEIPT.json'
    d=json.loads(old.read_text(encoding='utf-8-sig')); records=sorted([r for r in d['train']+d['heldout'] if r['camera_id']<48],key=lambda r:r['camera_id'])
    assert len(records)==48 and all(Path(r['rgb']).is_file() and Path(r['mask']).is_file() for r in records)
else:
    records=[]
    with h5py.File(main,'r') as m,h5py.File(ann,'r') as a:
        for cid in range(48):
            r,_=reader.export_view(m,a,cid,frame,folder,cid);records.append(r)
            print(code,cid,flush=True)
assert [r['camera_id'] for r in records]==list(range(48)) and all(r['frame']==frame for r in records)
receipt=dict(sample_id=sid,main_archive=str(main),annotation_archive=str(ann),source_view_ids=['02','11','17','26','35','41'],records=records,status='COMPLETE48',source_receipt=str(old) if code=='0097' else 'DIRECT_SMC_DECODE',new_model_calls=0)
save(folder/'READ48_RECEIPT.json',receipt)
protected={2,11,17,26,35,41}; held=[5,12,19,27,34,42] if code=='0097' else [1,8,12,23,28,33,38,44]
added=[]
if code=='0097':
    centers=np.array([np.array(r['official_RT_c2w'])[:3,3] for r in records]); vectors=centers-centers.mean(axis=0); vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    while len(held)<8:
        eligible=sorted(set(range(48))-protected-set(held))
        scores={i:min(float(np.arccos(np.clip(vectors[i]@vectors[j],-1,1))) for j in held) for i in eligible}
        chosen=max(eligible,key=lambda i:(scores[i],-i));held.append(chosen);added.append(dict(camera_id=chosen,min_angular_distance_degrees=float(np.degrees(scores[chosen]))))
assert not set(held)&protected
split=dict(sample_id=sid,train=sorted(set(range(48))-set(held)),heldout_sorted=sorted(held),source_view_ids=['02','11','17','26','35','41'],train_count=40,heldout_count=8,role='PROSPECTIVE_ONLY',historical_roles_unchanged=True,added_held_former_train=added,rule=('Six original 5MP HELD preserved; sequentially select maximum minimum angular distance of camera-center directions relative to mean of 48 centers, lowest physical ID on ties; exclude student sources; no image or result inspection.' if code=='0097' else '0019 physical held IDs, disjoint from student six and body cameras; frozen before fitting.'),historical_teacher_source_0097=[2,11,20,26,35,41] if code=='0097' else None)
save(folder/'SPLIT40_8.json',split)
print('DONE',sid,'held',sorted(held),'added',added,flush=True)
