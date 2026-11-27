"""External paths for the historical GW40 / DN_GW replay."""

import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path(__file__).with_name("runtime")


def external_path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


RESULTS = external_path("GW40_RESULTS_ROOT", REPO / "results")
OUTPUT = external_path("GW40_OUTPUT_ROOT", RESULTS / "DNA_0019_UNIFIED_PIPELINE_REPLAY_20261125")
ARTIFACTS = external_path("GW40_ARTIFACTS_ROOT", REPO / "artifacts/thuman20")
DATASETS = external_path("GW40_DATASET_ROOT", REPO / "datasets")
SMPLX_MODEL = external_path("GW40_SMPLX_MODEL", DATASETS / "smplx/SMPLX_NEUTRAL.npz")
VIT_PYTHON = external_path("GW40_VIT_PYTHON", sys.executable)
QA_PYTHON = external_path("GW40_QA_PYTHON", sys.executable)
COLMAP = os.environ.get("GW40_COLMAP", "colmap")
QA_FONT = external_path("GW40_QA_FONT", "C:/Windows/Fonts/arial.ttf")
MONO_FONT = external_path("GW40_MONO_FONT", "C:/Windows/Fonts/consola.ttf")
OFFICIAL = RESULTS / "DNA_GW_OFFICIAL_DIRECT_20261102"
RASTER = RESULTS / "DNA_GW_GEOMETRY_STUDENT_20261103/teacher_neighbor_fix"
VENDOR = external_path("GW40_VENDOR_ROOT", OFFICIAL / "vendor/GaussianWrapping")
TEACHER_PYTHON = external_path("GW40_TEACHER_PYTHON", OFFICIAL / "env/Scripts/python.exe")
RASTER_PYTHON = external_path("GW40_RASTER_PYTHON", RASTER / "env/Scripts/python.exe")
