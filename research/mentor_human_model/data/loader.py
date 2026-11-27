"""加载六视图推理输入；训练scan只在显式监督分支读取。"""

import json
from pathlib import Path

import numpy as np
import torch


def load_manifest(path, device="cpu", include_targets=True):
    """按清单返回输入、可选监督与冻结RGB特征的实际路径。"""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for item in manifest["samples"]:
        input_path = Path(item["input_npz"])
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        with np.load(input_path, allow_pickle=False) as arrays:
            batch = {key: torch.as_tensor(arrays[key], device=device)
                     for key in arrays.files}
        target = {}
        if include_targets:
            target_path = Path(item["target_npz"])
            if not target_path.is_absolute():
                target_path = path.parent / target_path
            with np.load(target_path, allow_pickle=False) as arrays:
                target = {key: torch.as_tensor(arrays[key], device=device)
                          for key in arrays.files}
        result.append({"id": item["id"], "batch": batch,
                       "target": target, "snapshots": item["snapshots"],
                       "lineage": item["lineage"]})
    return result
