"""绑定 0102 的原米制世界、相机分工、旧区域与传感器容差。"""
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import h5py
import numpy as np
from plyfile import PlyData


REPO = Path(__file__).resolve().parents[2]
from research.dna_gw40_dngw0019_replay_20261125.paths import DATASETS, RESULTS
OUT = RESULTS / "DNA_GW_SECOND_IDENTITY_0102_20261123"
RUN = OUT / "GW_0102_34BUILD_6AUX_8EVAL"
QA = OUT / "INDEPENDENT_QA_DATA"
FIG = OUT / "PPT_REPORT_FIGURES"
DATA = DATASETS / "dna_rendering"
KINECT = DATA / "dna_rendering_part1_kinect/0102_02_kinect.smc"
HIST = RESULTS / "DNA_NEUS2_CROSS_ID_AND_STUDENT_20261026"
OLD = HIST / "teachers/0102_02_0000"
CROSS = RESULTS / "DNA_DNGW_RESTORATION_AND_GEOMETRY_20261118/geometry_0102"
TAG = "DNA_RENDERING__0102__0102_02__0000/GIVEN_PRIOR"
ORIGINAL = REPO / "artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021/inputs" / TAG
U600 = HIST / "student_evaluation/U600_NEW_MISSING" / TAG
MODELS = dict(GW_OFFICIAL_FULL=RUN/"model/mesh_ours_2pivots_post_texture_refined_999.ply",
              GW_RAW_UNFILTERED=RUN/"model/RAW_UNFILTERED.ply",
              NEUS2_FULL=OLD/"formal/NATIVE_ALL_COMPONENTS_WORLD.ply",
              U600_FINAL=U600/"clothed_mesh.ply")
EVAL = [1,8,12,23,28,33,38,44]
AUX = [3,5,18,27,32,36]
SOURCE = [2,11,17,26,35,41]


def read(path):
    """读取已有结构化证据。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def original_mesh(path):
    """直接按原 PLY 行读三角顶点及面，返回值与原始 face_id 同序。"""
    data=PlyData.read(path,known_list_len={"face":{"vertex_indices":3,"vertex_index":3}})
    vertices=data["vertex"].data
    face_data=data["face"].data
    key="vertex_indices" if "vertex_indices" in face_data.dtype.names else "vertex_index"
    faces=face_data[key]
    if faces.ndim==1:
        faces=np.stack(faces)
    return np.column_stack([vertices[k] for k in "xyz"]).astype("f4"),np.ascontiguousarray(faces,dtype="u4")


def save(path, data):
    """写出 UTF-8 结构化证据。"""
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def table(path, rows):
    """保留缺测值和单位写出逐项记录。"""
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def frozen_boxes():
    """继承右鞋袍盒，按同帧旧人体高度带定义其余观察窗。"""
    with np.load(OLD/"source_fitted/SOURCE_FITTED.npz") as src:
        points=src["vertices_world"]
        joints=src["joints_world"]
    lo,hi=points.min(0),points.max(0)
    boxes={"RIGHT_SHOE_ROBE":read(CROSS/"INPUT_AND_PAIR_MANIFEST.json")["groups"][0]["roi"]["positions"]}
    for name,y0,y1 in read(OLD/"quality/NEUS2/FIGURE_INDEX.json")["local_bands"]:
        a,b=lo.copy(),hi.copy()
        a[1],b[1]=lo[1]+(hi[1]-lo[1])*y0,lo[1]+(hi[1]-lo[1])*y1
        boxes[name+"_HEIGHT_BAND"]=[a.tolist(),b.tolist()]
    a,b=lo.copy(),hi.copy()
    a[1]=min(joints[10,1],joints[11,1])-.1*abs(joints[11,1]-joints[9,1])
    boxes["LOWER_HEM_HEIGHT"]=[a.tolist(),b.tolist()]
    return boxes


def cameras():
    """将冻结角色表中的原标定绑定到物理相机。"""
    with (OUT/"0102_CAMERA_ROLE_MANIFEST.csv").open(encoding="utf-8-sig",newline="") as f:
        return {int(x["camera"]):x for x in csv.DictReader(f)}


def camera_frame(row):
    """从相应原容器准确解码该台 RGB、Mask、K、D、RT。"""
    with Path(row["native_rgb_smc"]).open("rb") as rb, Path(row["annotation_smc"]).open("rb") as ab, h5py.File(rb,"r") as rgb, h5py.File(ab,"r") as ann:
        image=cv2.imdecode(np.asarray(rgb[row["rgb_key"]]),cv2.IMREAD_COLOR)
        mask=cv2.imdecode(np.asarray(ann[row["mask_key"]]),cv2.IMREAD_UNCHANGED)
        c=ann[f'Camera_Parameter/{int(row["camera"]):02}']
        k,d,rt=[np.asarray(c[x],np.float64) for x in ("K","D","RT")]
    max_channel=mask if mask.ndim==2 else np.max(mask,axis=2)
    gray=mask if mask.ndim==2 else cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ref=max_channel>0
    counts=dict(channels=1 if mask.ndim==2 else mask.shape[2],height=mask.shape[0],width=mask.shape[1],
                min_value=int(mask.min()),max_value=int(mask.max()),
                nonbinary_pixels=int(((max_channel!=0)&(max_channel!=255)).sum()),
                nonbinary_channel_elements=int(((mask!=0)&(mask!=255)).sum()),
                differing_color_channel_pixels=0 if mask.ndim==2 else int((mask.min(2)!=max_channel).sum()),
                any_positive=int(ref.sum()),any_at_least_128=int((max_channel>=128).sum()),
                positive_vs_ge128=int((ref^(max_channel>=128)).sum()),
                gray_positive_diff=int((ref^(gray>0)).sum()))
    return image,ref,k,d,rt,counts


def grid_band():
    """按本帧原学生世界立方体复用一格宽度。"""
    with np.load(ORIGINAL/"INPUTS.npz") as z:
        return z["unit_scale"].item()/128


def project(points,k,d,rt):
    """把原米制世界点投影至原传感器的含畸变像素。"""
    cam=(points-rt[:3,3])@rt[:3,:3]
    uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
    return uv,cam[:,2]


def project_box(bounds,k,d,rt,h,w):
    """将固定世界盒八角投影为同一像素观察窗。"""
    corners=np.array(list(product(*zip(*bounds))),dtype=np.float64)
    uv,_=project(corners,k,d,rt)
    a=np.maximum(np.floor(uv.min(0)).astype(int),0)
    b=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w,h])
    return [int(v) for v in [*a,*b]]


def freeze_protocol():
    """记录候选冻结前的比较定义与原证据路径。"""
    QA.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(exist_ok=True)
    save(QA/"FROZEN_QA_PROTOCOL.json",dict(frame="DNA_RENDERING/0102/0102_02/0000",EVAL=EVAL,AUX=AUX,
         source=SOURCE,shared_world="original DNA metric world; identity transform",models={k:str(v) for k,v in MODELS.items()},
         old_frozen_box=str(CROSS/"INPUT_AND_PAIR_MANIFEST.json"),prior=str(OLD/"source_fitted/SOURCE_FITTED.npz"),
         region_boxes=frozen_boxes(),region_meaning="overlapping observation windows, not material labels",
         raw_tolerance_m=grid_band(),raw_tolerance_source=str(REPO/"_dna_shared_clothing_query_aux/cache_fixed_native_diagnostic.py"),
         native_kinect="8 original sensor positive depth and native mask; stable 3x3 positive with peak-to-peak <= one 0102 grid",
         residual="first hit camera-Z minus raw native camera-Z in millimetres; absolute MAE/median/P95 include all positive first hits",
         signed_states="stable and hit: near below -band, compatible within band, far above +band; no hit separate; unstable classified unknown",
         image_metric="full distorted 2048x2448, native pixel centres, reference any original channel >0; TP FN FP TN computed for full and each projected window",
         decision="same visible material requires independent native or AUX evidence; invalid or boundary has no physical layer label",
         old_NeuS2_training_aux_overlap=AUX,old_NeuS2_held=EVAL,
         prepared_without_EVAL_images=True))


if __name__=="__main__":
    freeze_protocol()
