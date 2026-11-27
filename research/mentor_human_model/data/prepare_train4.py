"""复用真实六源THuman输入，独立生成合法训练scan监督。"""

import argparse
import json
from pathlib import Path

import numpy as np

from research.dna_gw40_dngw0019_replay_20261125.paths import DATASETS
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "artifacts/thuman20/stage47R_r25_production_cpu_smoke/body_stabilize/strict_fusion_clothing_exit_audit_20261003/LONG_TERM_VGGT_20261006/AUTHORITATIVE_RGB_INPUTS/CACHE_BINDINGS.json"
TRAIN_IDS = ("0000", "0003", "0004", "0007")
TH8 = ("0418", "0419", "0420", "0421", "0423", "0424", "0425", "0430")


def read_obj_geometry(path):
    """读取本数据集OBJ几何，避免监督生产加载无关的巨大纹理。"""
    vertices, faces = [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(x.split("/")[0])-1 for x in line.split()[1:]]
                for j in range(1,len(indices)-1):
                    faces.append([indices[0],indices[j],indices[j+1]])
    return np.asarray(vertices,dtype=np.float32),np.asarray(faces,dtype=np.int64)


def sample_scan(vertices, faces, count, seed):
    """按三角面积采样完整scan，法线由同一变换后的面方向给出。"""
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=-1)
    choices = rng.choice(len(faces), count, p=area2 / area2.sum())
    uv = rng.random((count, 2))
    su = np.sqrt(uv[:, :1])
    weights = np.concatenate((1-su, su*(1-uv[:, 1:]), su*uv[:, 1:]), axis=-1)
    points = (triangles[choices] * weights[..., None]).sum(1)
    normals = cross[choices] / area2[choices, None]
    return points.astype(np.float32), normals.astype(np.float32)


def query_camera(camera_K, camera_E, yaw, target):
    """只从已知源相机的轨道定义非源查询相机。"""
    centers = -np.einsum("vji,vj->vi", camera_E[:, :3, :3], camera_E[:, :3, 3])
    orbit_center = centers.mean(0)
    radius = np.linalg.norm((centers - orbit_center)[:, :2], axis=-1).mean()
    angle = np.deg2rad(yaw)
    eye = orbit_center + np.array([radius*np.sin(angle), -radius*np.cos(angle), 0])
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    extrinsic = np.eye(4); extrinsic[:3, :3] = rotation; extrinsic[:3, 3] = -rotation @ eye
    return {"yaw": float(yaw), "K": camera_K[0].tolist(), "E": extrinsic.tolist()}


def prepare(output, scan_root, count):
    """固定四合法训练身份；输入归一化仅取输入SMPL-X，不读scan统计。"""
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads(LEGACY.read_text(encoding="utf-8"))
    formal = json.loads((ROOT/"manifests/thuman20/splits_formal_4to1.json").read_text(encoding="utf-8"))
    samples = []
    contact = Image.new("RGB", (6*259, 4*279), "white")
    draw = ImageDraw.Draw(contact)
    for index, subject in enumerate(TRAIN_IDS):
        if subject not in formal["train"] or subject in TH8 or subject in formal["test"]:
            raise ValueError(f"非合法训练身份: {subject}")
        sid = f"THUMAN/{subject}/static/{subject}"
        source = Path(old["source_paths"][sid])
        with np.load(source, allow_pickle=False) as z:
            xyz = z["smplx_xyz"].astype(np.float32)
            center = (xyz.min(0)+xyz.max(0))*.5
            scale = np.float32(np.ptp(xyz, axis=0).max()*1.3)
            E = np.broadcast_to(np.eye(4, dtype=np.float32), (6,4,4)).copy()
            E[:, :3, :3] = z["R"]; E[:, :3, 3] = z["t"]
            inputs = dict(images=z["rgb"][None], masks=z["mask"][:,0][None], K=z["K"][None], E=E[None],
                          smpl_vertices=xyz[None], smpl_faces=z["smplx_faces"], smpl_canonical=z["smplx_xyz_canonical"][None],
                          smpl_normals=z["smplx_normals_canonical"][None], unit_center=center[None], unit_scale=np.asarray([scale]))
            targets = dict(depth=z["target_depth"][:,0][None], depth_valid=z["target_depth_valid"][:,0][None],
                           world_normal=np.moveaxis(z["target_normal"],1,-1)[None], normal_valid=z["target_normal_valid"][:,0][None])
            lineage = json.loads(str(z["source_lineage_json"]))
            for view in range(6):
                img = Image.fromarray(np.clip(np.moveaxis(z["rgb"][view],0,-1)*255,0,255).astype(np.uint8)).resize((259,259))
                contact.paste(img,(view*259,index*279+20))
        draw.text((0,index*279),f"TRAIN {subject}; six source RGB; fit observation, not blind test", fill="black")
        folder = output / subject; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/"INPUTS.npz", **inputs)
        scan_path = scan_root/subject/f"{subject}.obj"
        raw_vertices, raw_faces = read_obj_geometry(scan_path)
        # Blender实际OBJ导入绕X转90度；旧host bbox预览未体现Y负号。
        vertices = raw_vertices[:,[0,2,1]] * np.asarray([1,-1,1],dtype=np.float32)
        faces = raw_faces
        points, normals = sample_scan(vertices,faces,count,20261016+index)
        targets.update(scan_points=points[None],scan_normals=normals[None])
        np.savez_compressed(folder/"TARGETS.npz", **targets)
        np.savez_compressed(folder/"ORACLE_SCAN.npz", vertices=vertices, faces=faces, points=points,normals=normals)
        transform = dict(center=center.tolist(), scale=float(scale), source="source_known_camera_smplx_bbox_1.3_margin", isotropic=True)
        (folder/"INPUT_TRANSFORM.json").write_text(json.dumps(transform,indent=2),encoding="utf-8")
        cameras = [query_camera(inputs['K'][0],E,y,np.zeros(3)) for y in range(30,390,10)]
        # 查询只写相机；30/90/...与六源yaw不同。
        (folder/"QUERY_CAMERAS.json").write_text(json.dumps({"cameras":cameras},indent=2),encoding="utf-8")
        lineage.update(input_source=str(source), scan_source=str(scan_path), scan_role="supervision_and_oracle_only",
                       scan_axis_rule="raw_xyz -> world_(x,-z,y); proper_rotation_keep_face_winding", normalization=transform,
                       fit_role="six_source_RGB_YOLO_COCO17_and_known_camera_MultiviewSMPLifyX; no_scan_fit", prior_pretraining_overlap="unknown",
                       smplx_fit_producer=str(Path(lineage['accepted_prior_geometry']).parent.parent/'A/producer_manifest.json'),
                       previously_used_diagnostic=subject in ("0004","0007"))
        samples.append(dict(id=sid,input_npz=f"{subject}/INPUTS.npz",target_npz=f"{subject}/TARGETS.npz",
                            oracle_npz=f"{subject}/ORACLE_SCAN.npz",query_cameras=f"{subject}/QUERY_CAMERAS.json",
                            snapshots=old["cache_paths"][sid],lineage=lineage))
    manifest = dict(protocol="six_source_calibrated_THuman_train4_shared_model_20261016", train_ids=list(TRAIN_IDS),
                    heldout_TH8=list(TH8), final_test_excluded=formal["test"], samples=samples,
                    note="4/7 known diagnostic train identities; no final-test loading; input NPZ has no target arrays")
    (output/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    contact.save(output/"TRAIN4_SOURCE_RGB.jpg")
    print(json.dumps({"output":str(output),"train_ids":list(TRAIN_IDS),"source_views":6,"scan_points_per_id":count},ensure_ascii=False))


def main():
    """生成真实数据合同，不运行训练。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"research/mentor_human_model/results/TRAIN4_DATA")
    parser.add_argument("--scan-root",type=Path,default=DATASETS/"THuman2.0/THuman2.0")
    parser.add_argument("--scan-points",type=int,default=65536)
    args=parser.parse_args();prepare(args.output,args.scan_root,args.scan_points)


if __name__ == "__main__":
    main()
