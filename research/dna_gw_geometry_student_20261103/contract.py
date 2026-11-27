"""从原 28 样本与固定 V 冻结三支共用训练条件。"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from research.dna_2dgs_body_depth_student_20261102.contract import read, SID
from research.dna_b_adapter_20261021.paired_train import write_json


from research.dna_gw40_dngw0019_replay_20261125.paths import RESULTS
OUT = RESULTS / 'DNA_GW_GEOMETRY_STUDENT_20261103' / 'student'
OLD = RESULTS / 'DNA_2DGS_BODY_DEPTH_STUDENT_20261102' / 'TRAINING_CONTRACT.json'
QA = RESULTS / 'DNA_GW_CLEAN_QA_AND_STUDENT_GATE_20261103'
V_PATH = QA / 'candidate_support/GW_CANDIDATE_VALID_PRE_QA.npz'
V_RECEIPT = QA / 'candidate_support/GW_VALID_PRE_QA_FREEZE.json'
BRANCHES = {'C_GW': (0., 0.), 'D_GW': (1., 0.), 'DN_GW': (1., .05)}


def draw_plan(samples):
    """保留 manifest 内顺序，固定第一 DNA，循环第二 DNA 与两条 TH。"""
    dna = [row['id'] for row in samples if row['domain'] == 'DNA' and row['id'] != SID]
    th = [row['id'] for row in samples if row['domain'] == 'TH']
    if len(set(dna)) != 23 or len(set(th)) != 4 or len(samples) != 28:
        raise ValueError('原 24 DNA 与 4 TH 样本不符')
    return [
        dict(update=step + 1, DNA=[SID, dna[step % len(dna)]],
             TH=[th[(2 * step + offset) % len(th)] for offset in range(2)])
        for step in range(200)
    ]


def prepare():
    """固定计划和原输入支持，给教师冻结留独立收据。"""
    old = read(OLD)
    plan = draw_plan(old['samples'])
    counts = Counter(sid for row in plan for domain in ('DNA', 'TH') for sid in row[domain])
    sample = next(row for row in old['samples'] if row['id'] == SID)
    vr = read(V_RECEIPT)
    with (np.load(V_PATH) as v, np.load(sample['input_npz']) as inp,
          np.load(sample['target_npz']) as target):
        kval = target['depth_valid'].astype(bool)
        pixels = np.argwhere(v['valid'])
        observed = v['valid'].sum((0, 2, 3)).tolist()
        for key in ('K', 'E', 'unit_center', 'unit_scale'):
            if not np.array_equal(inp[key], v[key]):
                raise ValueError(f'固定 V 的 {key} 与原 INPUTS 不同')
        if (not np.array_equal(pixels, v['bvyx']) or len(pixels) != 50145
                or observed != vr['per_source_count'] or int(kval.sum()) != 70738
                or (v['valid'] & ~kval).any()
                or [int(cam) for cam in sample['source_view_ids']] != v['source_camera_ids'].tolist()):
            raise ValueError('固定 V 与原 Kinect 像素合同不符')
        scale = inp['unit_scale'].tolist()
    common = ('initialize_from', 'optimizer_options', 'optimizer_initialization',
              'rng_initialization', 'samples', 'sample_weight', 'frozen', 'camera_head',
              'student_prior', 'updates_per_branch')
    contract = {key: old[key] for key in common}
    contract.update(
        scope='AUTHORIZED_GW_GEOMETRY_3X200', output=str(OUT), common_contract=str(OLD),
        branches={branch: dict(depth=weights[0], normal=weights[1]) for branch, weights in BRANCHES.items()},
        checkpoints=[0, 100, 200], total_update_budget=600, draw_plan=plan,
        draw_policy='original manifest order, fixed first DNA, cycle remaining DNA and original TH pairs',
        exposures_per_branch=dict(counts), teacher_exposures={'C_GW': 0, 'D_GW': 200, 'DN_GW': 200},
        source_V=str(V_PATH), source_V_receipt=str(V_RECEIPT), teacher_depth_count=50145,
        teacher_depth_per_source=observed, original_Kinect_count=70738, unit_scale=scale,
        camera17='MASK_SEMANTICS_UNRESOLVED; original Kinect retained',
        camera41='Original Kinect retained; outside new fixed V',
        normal_definition='Signed cosine on independently frozen VN subset of V',
        new_DNA_field=False, E1=False, certified_GT=False,
        teacher_target_receipt=str(OUT / 'teacher_target/RECEIPT.json'),
        teacher_target_archive=str(OUT / 'teacher_target/TARGETS.npz'),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'TRAINING_CONTRACT.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, contract)
    write_json(OUT / 'DRAW_PLAN.json', plan)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
