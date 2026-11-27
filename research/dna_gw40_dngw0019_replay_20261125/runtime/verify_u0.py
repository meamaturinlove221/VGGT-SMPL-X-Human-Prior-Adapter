"""核对三份实际 U0 与 U600 的全部保存模型、优化器和 RNG。"""

import argparse
from pathlib import Path

import numpy as np
import torch

from research.dna_gw40_dngw0019_replay_20261125.paths import REPO as ROOT
from research.dna_b_adapter_20261021.paired_train import write_json
from research.dna_gw_geometry_student_20261103.contract import BRANCHES, OUT, read


def equal(left, right):
    """逐值比较原检查点实际使用的张量、数组及嵌套状态。"""
    if torch.is_tensor(left):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) == type(right) and len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right))
    return left == right


def main(branches):
    """实际逐值一致时，给相同原入口 U0 回用已存评价的依据。"""
    torch.set_num_threads(4)
    contract = read(OUT / 'TRAINING_CONTRACT.json')
    path = contract['initialize_from']
    parent = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    rows = {}
    for branch in branches:
        checkpoint = OUT / branch / 'checkpoint_u0000.pt'
        actual = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        config = {key: parent['config'][key] for key in actual['config']}
        for key in ('base_weights', 'ft_checkpoint'):
            config[key] = str((ROOT / config[key]).resolve())
        flags = {key: equal(actual[key], parent[key]) for key in ('model', 'optimizer', 'rng')}
        flags['config'] = equal(actual['config'], config)
        rows[branch] = dict(checkpoint=str(checkpoint), exact_state=flags,
                            parent_update=parent['update'], actual_update=actual['update'])
    status = 'MATCHED_ORIGINAL_U600' if all(all(row['exact_state'].values()) for row in rows.values()) else 'DIFFERENCE'
    receipt = 'ACTUAL_U0_COMPARISON.json' if len(branches) == len(BRANCHES) else f'ACTUAL_U0_COMPARISON_{"_".join(branches)}.json'
    write_json(OUT / receipt, dict(status=status, original=path, branches=rows))
    if status == 'DIFFERENCE':
        raise ValueError('实际 U0 与 U600 有差异')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=BRANCHES, action='append')
    args = parser.parse_args()
    main(args.branch or BRANCHES)
