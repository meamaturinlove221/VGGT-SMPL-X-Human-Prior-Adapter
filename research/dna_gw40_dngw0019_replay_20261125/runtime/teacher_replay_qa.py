"""Evaluate the frozen GW40 full and BODY_ONLY meshes on original observations."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import trimesh

from research.dna_gw40_dngw0019_replay_20261125.paths import ARTIFACTS, QA_FONT, REPO as ROOT, RESULTS, SMPLX_MODEL as SMPL
OUT = RESULTS / "DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125"
OLD = RESULTS / "DNA_GW_OFFICIAL_DIRECT_20261102"
OLD_INPUT = RESULTS / "DNA_GAUSSIAN_TEACHER_EXEC_20261102/inputs/DNA0019_INPUT.json"
OLD_HELD = RESULTS / "DNA_GW_CLEAN_QA_AND_STUDENT_GATE_20261103/clean_independent_qa/HELD8_MICRO.csv"
OLD_KINECT = RESULTS / "DNA_MENTOR_GEOMETRY_REPAIR_20261114/gw_body_qa/GW_BODY_ONLY_QA.json"
BODY_IDS = (2, 26, 11, 35)
SOURCE_IDS = (2, 11, 17, 26, 35, 41)
THREADS = 4
GROUPS = {
    "HEAD": (12, 15, 22, 23, 24),
    "LEFT_HAND": (20, *range(25, 40)),
    "RIGHT_HAND": (21, *range(40, 55)),
    "LEGS": (1, 2, 4, 5, 7, 8),
    "SHOES": (10, 11),
}
FONT = ImageFont.truetype(str(QA_FONT), 20)

sys.path[:0] = [
    str(ROOT / "research/dna_gw_second_identity_0102_20261123"),
]
from body_only import supported
from extract_gw_human_display import project, save_ply
from independent_rays import pixel_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def png(path, flags):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def input_contract(case):
    sequence = case + "_01"
    tag = f"DNA_RENDERING__{case}__{sequence}__0000"
    original = ARTIFACTS / "DNA_MULTISUBJECT_REPAIR_20261024/inputs" / tag
    split_path = original / "SPLIT40_8.json"
    receipt_path = next(original.glob("*READ48_RECEIPT.json"))
    split, receipt = read(split_path), read(receipt_path)
    source, held = split["train"], split["heldout_sorted"]
    if len(source) != 40 or len(held) != 8 or set(source) & set(held) or set(source + held) != set(range(48)):
        raise ValueError("Original SPLIT40_8 does not bind 48 disjoint physical cameras")
    if not set(BODY_IDS) <= set(source) or tuple(map(int, split["source_view_ids"])) != SOURCE_IDS:
        raise ValueError("Original teacher and student camera roles differ from the frozen recipe")
    rows = {r["camera_id"]: r for r in receipt["records"]}
    if set(rows) != set(range(48)) or {r["frame"] for r in rows.values()} != {0}:
        raise ValueError("Prepared RGB/mask records do not bind the same 48 cameras and frame 0000")
    given = ARTIFACTS / "DNA_INTEGRATED_REPAIR_20261021/inputs" / tag / "GIVEN_PRIOR"
    lineage = read(given / "LINEAGE.json")
    kinect = lineage["source_lineage"]["depth_source"].split("::", 1)[0]
    body = read(OLD / "human_body_only/BODY_ONLY_RECEIPT.json")
    old = read(OLD_INPUT)
    if [r["camera_id"] for r in body["selection"]] != list(BODY_IDS):
        raise ValueError("Historical BODY_ONLY camera sequence changed")
    with np.load(given / "INPUTS.npz") as z, np.load(SMPL) as model:
        prior = z["smpl_vertices"][0]
        if not np.array_equal(z["smpl_faces"], model["f"]) and not np.array_equal(z["smpl_faces"][0], model["f"]):
            raise ValueError("SMPL-X visual crop topology cannot be transferred to these vertices")
        parts = model["weights"].argmax(1)
        band_m = float(z["unit_scale"].item()) / 128
    boxes = {"SUBJECT": np.stack([prior.min(0) - .10, prior.max(0) + .10]).tolist()}
    for name, joints in GROUPS.items():
        selected = prior[np.isin(parts, joints)]
        boxes[name] = np.stack([selected.min(0) - .06, selected.max(0) + .06]).tolist()
    contract = dict(
        sample=split["sample_id"], split_receipt=str(split_path), prepared_receipt=str(receipt_path),
        train=source, held=held, six_student_sources=list(SOURCE_IDS), body_train=list(BODY_IDS),
        historical_train=old["train_ids"], historical_held=old["held_ids"], historical_input=str(OLD_INPUT),
        historical_body_receipt=str(OLD / "human_body_only/BODY_ONLY_RECEIPT.json"),
        body_filter=body["rule"], kinect_smc=kinect, region_boxes_world_m=boxes,
        original_inputs=str(given / "INPUTS.npz"), original_targets=str(given / "TARGETS.npz"),
        crop_source=f"{given / 'INPUTS.npz'} SMPL-X world vertices with {SMPL} dominant weights",
        held_definition="Prepared 1224x1024 grayscale mask >0, direct same-frame K/E, full integer pixel-center rays",
        kinect_definition="All eight unfiltered native positive mask/depth sensors; reliable 3x3 positive span <= unit_scale/128; Euclidean first range and camera Z",
        kinect_band_m=band_m, historical_held_scores=str(OLD_HELD), historical_native_scores=str(OLD_KINECT),
        historical_native_scope="Previously filtered native sample cohort; 10.218 mm equal-sensor first range MAE is context",
        crops_scope="Visual inspection windows, no anatomy ground truth or admission threshold",
    )
    for cam in BODY_IDS + tuple(held):
        row = rows[cam]
        mask = png(row["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (1024, 1224) or not np.any(mask >= 128):
            raise ValueError(f"No original foreground on prepared camera {cam:02d}")
    return contract, rows


def read_ply(path):
    data = PlyData.read(path)
    vertices = data["vertex"].data
    xyz = np.column_stack([vertices[k] for k in "xyz"]).astype("f4")
    rgb = np.column_stack([vertices[k] for k in ("red", "green", "blue")]).astype("u1")
    faces = np.stack(data["face"].data["vertex_indices"]).astype("u4") if len(data["face"].data) else np.empty((0, 3), "u4")
    return xyz, faces, rgb


def mesh_scene(mesh):
    vertices, faces, _ = mesh
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Invalid metric mesh")
    scene = o3d.t.geometry.RaycastingScene(nthreads=THREADS)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    return scene


def prove_body(rows, run, parent, body, point_path):
    folder = run / "human_body_only"
    receipt_path = folder / "BODY_ONLY_RECEIPT.json"
    report = read(receipt_path)
    selection = report.get("selection", [])
    declared = report.get("body_mask_ids", [r["camera_id"] for r in selection])
    if declared != list(BODY_IDS):
        raise ValueError("BODY_ONLY consumed the wrong support cameras")
    if "body_mask_ids" in report:
        declared_rule = tuple(report.get(k) for k in ("threshold", "dilation_square", "votes", "positions", "held_used"))
        if declared_rule != (128, 17, 3, 7, 0):
            raise ValueError("BODY_ONLY receipt reports a different filter or held-out exposure")
    for actual in selection:
        expected = rows[actual["camera_id"]]
        for key in ("rgb", "mask", "K", "E"):
            if actual[key] != expected[key]:
                raise ValueError(f"BODY_ONLY {key} differs for camera {actual['camera_id']}")
    selected = [rows[cam].copy() for cam in BODY_IDS]
    for row in selected:
        mask = np.asarray(Image.open(row["mask"]).convert("L")) >= 128
        row["_mask"] = cv2.dilate(mask.astype("u1"), np.ones((17, 17), "u1")) > 0
    v, f, rgb = parent
    triangle = v[f].astype("f8")
    keep = supported(v, selected)[f].all(1) & supported(triangle.mean(1), selected)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        keep &= supported((triangle[:, a] + triangle[:, b]) / 2, selected)
    face_ids = np.load(folder / "ORIGINAL_FACE_INDICES.npy")
    vertex_ids = np.load(folder / "ORIGINAL_VERTEX_INDICES.npy")
    removed = np.load(folder / "REMOVED_FACE_INDICES.npy")
    expected_faces = np.flatnonzero(keep)
    valid = (np.array_equal(face_ids, expected_faces) and np.array_equal(removed, np.flatnonzero(~keep))
             and np.array_equal(vertex_ids, np.unique(f[keep]))
             and np.array_equal(body[0], v[vertex_ids]) and np.array_equal(body[2], rgb[vertex_ids])
             and np.array_equal(vertex_ids[body[1]], f[face_ids]))
    points = read_ply(point_path)
    valid &= not len(points[1]) and np.array_equal(points[0], body[0]) and np.array_equal(points[2], body[2])
    if not valid:
        raise ValueError("BODY_ONLY fails exact original 7-sample face selection or XYZ/RGB/face/point readback")
    return dict(body_receipt=str(receipt_path), body_original_faces=len(face_ids),
                deleted_original_faces=len(removed), point_world_xyz_rgb_equal=True,
                triangle_original_mapping_equal=True, train_only_mask_inputs=list(BODY_IDS),
                same_frame_rgb_mask_K_E=[{k: row[k] for k in ("camera_id", "rgb", "mask", "K", "E")} for row in selected])


def cast(scene, origin, direction, mesh=None):
    rays = np.empty((*direction.shape[:-1], 6), "f4")
    rays[..., :3], rays[..., 3:] = origin, direction
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=THREADS)
    depth = hit["t_hit"].numpy()
    valid = np.isfinite(depth)
    face = np.where(valid, hit["primitive_ids"].numpy(), -1).astype("i4")
    rgb = None
    if mesh is not None:
        rgb = np.full((*depth.shape, 3), 245, "u1")
        uv = hit["primitive_uvs"].numpy()[valid]
        weights = np.column_stack([1 - uv.sum(1), uv])
        rgb[valid] = np.rint(np.einsum("ni,nic->nc", weights, mesh[2][mesh[1][face[valid]]])).astype("u1")
    return np.where(valid, depth, np.nan), face, rgb


def raster(scene, mesh, row, shape):
    h, w = shape
    y, x = np.indices(shape, dtype="f4")
    grid = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(np.asarray(row["K"], "f4")).T
    rt = np.linalg.inv(np.asarray(row["E"], "f4"))
    return cast(scene, rt[:3, 3], grid @ rt[:3, :3].T, mesh)


def crop_box(bounds, row, shape):
    corners = np.array(list(product(*zip(*bounds))))
    projected, _ = project(corners, row)
    lo = np.maximum(np.floor(projected.min(0)), 0).astype(int)
    hi = np.minimum(np.ceil(projected.max(0)) + 1, shape[::-1]).astype(int)
    return tuple([*lo, *hi])


def panels(path, names, groups, heading, cell=(305, 245)):
    w, h = cell
    sheet = Image.new("RGB", (w * len(names), 37 + h * len(groups)), "white")
    caption = ImageDraw.Draw(sheet)
    caption.text((10, 7), heading, font=FONT, fill="black")
    for i, (region, images) in enumerate(groups):
        y = 37 + h * i
        for j, (label, src) in enumerate(zip(names, images)):
            image = src.copy() if isinstance(src, Image.Image) else Image.fromarray(src)
            image.thumbnail((w - 8, h - 31))
            x = w * j
            caption.text((x + 4, y + 2), f"{region} {label}", font=FONT, fill="black")
            sheet.paste(image, (x + (w - image.width)//2, y + 28))
    sheet.save(path)


def contour_overlay(image, ref, pred):
    result = image.copy()
    for binary, color in ((ref, (21, 225, 44)), (pred, (250, 42, 60))):
        contours, _ = cv2.findContours(binary.astype("u1"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    return result


def heldout(contract, rows, scenes, meshes, output):
    metrics, overview = [], []
    for cam in contract["held"]:
        row = rows[cam]
        rgb = png(row["rgb"], cv2.IMREAD_COLOR)[..., ::-1].copy()
        mask = png(row["mask"], cv2.IMREAD_GRAYSCALE) > 0
        shape = mask.shape
        buffers = {k: raster(scene, meshes[k], row, shape) for k, scene in scenes.items()}
        windows = {"FULL": (0, 0, shape[1], shape[0])}
        windows.update({name: crop_box(bounds, row, shape) for name, bounds in contract["region_boxes_world_m"].items()})
        images = [rgb]
        for name, (depth, face, color) in buffers.items():
            hit = np.isfinite(depth)
            images += [color, contour_overlay(rgb, mask, hit)]
            for label, crop in windows.items():
                metrics.append(dict(candidate=name, device=cam, region=label, **pixel_metrics(mask, hit, crop)))
        y, x = np.nonzero(mask)
        observed = (max(0, x.min() - 23), max(0, y.min() - 23), min(shape[1], x.max() + 24), min(shape[0], y.max() + 24))
        visual_windows = {**windows, "SUBJECT": observed}
        names = ("Original", "Full mesh", "Full contours", "BODY mesh", "BODY contours")
        groups = [(region, [Image.fromarray(image).crop(box) for image in images]) for region, box in visual_windows.items()]
        panels(output / f"HELD{cam:02d}_FULL_AND_REGIONS.jpg", names, groups, f"HELD {cam:02d}   {contract['sample']}   green observed, red predicted")
        overview.append((f"HELD {cam:02d}", [Image.fromarray(images[j]).crop(observed) for j in (0, 1, 3, 4)]))
        packed = {"ref_mask": mask, "K": row["K"], "E": row["E"]}
        for name, (depth, face, _) in buffers.items():
            packed.update({name+"_z": depth, name+"_face": face})
        np.savez_compressed(output / f"HELD{cam:02d}_FIRST_SURFACE.npz", **packed)
        print(f"HELD {cam:02d} complete", flush=True)
    for name in scenes:
        for label in windows:
            group = [r for r in metrics if r["candidate"] == name and r["region"] == label]
            vals = {k: sum(r[k] for r in group) for k in ("pixels", "TP", "FN", "FP", "TN")}
            t, n, p = (vals[k] for k in ("TP", "FN", "FP"))
            metrics.append(dict(candidate=name, device="ALL", region=label, **vals, IoU=t/(t+n+p) if t+n+p else None,
                                precision=t/(t+p) if t+p else None, recall=t/(t+n) if t+n else None))
    table(output / "HELD8_SILHOUETTE.csv", metrics)
    for first in (0, 4):
        panels(output / f"HELD8_SUBJECT_OVERVIEW_{first//4+1}.jpg", ("Original", "Full mesh", "BODY mesh", "BODY contours"),
               overview[first:first+4], f"{contract['sample']}   fixed views   green observed, red predicted", (255, 267))
    return {r["candidate"]: r for r in metrics if r["device"] == "ALL" and r["region"] == "FULL"}


def errors(dz, region, band):
    finite = np.isfinite(dz)
    hit = finite & region
    err = np.abs(dz[hit]) * 1000
    return dict(reference=int(region.sum()), hit=int(hit.sum()), miss=int((region & ~finite).sum()),
                MAE_mm=float(err.mean()) if len(err) else None, p95_mm=float(np.quantile(err, .95)) if len(err) else None,
                near=int((hit & (dz < -band)).sum()), far=int((hit & (dz > band)).sum()),
                within=int((hit & (np.abs(dz) <= band)).sum()))


def native(contract, scenes, output):
    band = contract["kinect_band_m"]
    frame = int(contract["sample"].rsplit("/", 1)[-1])
    metrics, parts, visuals = [], [], []
    kernel = np.ones((3, 3), "u1")
    with Path(contract["kinect_smc"]).open("rb") as stream, h5py.File(stream, "r") as h5:
        for dev in range(8):
            sensor = h5[f"Kinect/{dev}"]
            raw = sensor[f"depth/{frame}"][...]
            mask = cv2.imdecode(sensor[f"mask/{frame}"][...], cv2.IMREAD_UNCHANGED)
            gray = mask if mask.ndim == 2 else mask.max(2)
            y, x = np.nonzero((raw > 0) & (gray > 0))
            xy = np.column_stack([x, y])
            calib = h5[f"Calibration/Kinect/{dev:02d}"]
            K, D, RT = [np.asarray(calib[key], "f8") for key in ("K", "D", "RT")]
            normalized = cv2.undistortPointsIter(xy[:, None].astype("f8"), K, D, None, None,
                                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))[:, 0]
            d = np.column_stack([normalized, np.ones(len(xy))])
            reproj = cv2.projectPoints(d, np.zeros(3), np.zeros(3), K, D)[0][:, 0]
            valid = np.linalg.norm(reproj - xy, axis=1) <= .01
            dirs = d @ RT[:3, :3].T
            stable_3x3 = cv2.erode((raw > 0).astype("u1"), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            span = cv2.dilate(raw, kernel).astype(int) - cv2.erode(raw, kernel).astype(int)
            stable = valid & stable_3x3[y, x] & (span[y, x] <= band * 1000)
            obs_z = raw[y, x].astype("f8") * .001
            z_to_range = np.linalg.norm(dirs, axis=1)
            obs_range = obs_z * z_to_range
            world = RT[:3, 3] + dirs * obs_z[:, None]
            groups = {"ALL_POSITIVE": valid, "RELIABLE": stable}
            groups.update({name: stable & ((world >= bounds[0]) & (world <= bounds[1])).all(1)
                           for name, bounds in contract["region_boxes_world_m"].items()})
            packed = dict(pixel_xy=xy, depth_raw_mm=raw[y, x], observed_range_m=obs_range, stable=stable,
                          raw_world_m=world, calibrated_ray_valid=valid, K=K, D=D, RT=RT)
            residuals = {}
            images = []
            for name, scene in scenes.items():
                first, face, _ = cast(scene, RT[:3, 3], dirs)
                packed.update({name+"_first_z_m": first, name+"_first_face": face})
                residuals[name] = (first - obs_z) * z_to_range
                delta = residuals[name]
                color = np.full((*raw.shape, 3), 247, "u1")
                color[y, x] = [173, 173, 173]
                for mask_hit, value in ((~np.isfinite(delta), [227, 155, 45]), (delta < -band, [210, 47, 53]),
                                        (delta > band, [55, 91, 214]), (np.abs(delta) <= band, [44, 155, 86])):
                    choose = stable & mask_hit
                    color[y[choose], x[choose]] = value
                images.append(color[max(0,y.min()-8):y.max()+9, max(0,x.min()-8):x.max()+9])
                for region, subset in groups.items():
                    metrics.append(dict(candidate=name, device=dev, region=region, **errors(residuals[name], subset, band)))
            parts.append((residuals, groups))
            visuals.append((f"Kinect {dev}", images))
            np.savez_compressed(output / f"KINECT{dev}_FIRST_SURFACE.npz", **packed)
    for name in scenes:
        dz = np.concatenate([values[name] for values, _ in parts])
        for region in groups:
            chosen = np.concatenate([mm[region] for _, mm in parts])
            metrics.append(dict(candidate=name, device="ALL", region=region, **errors(dz, chosen, band)))
    pair = []
    for dev, (values, subset) in enumerate(parts):
        full, body = values["FULL"], values["BODY"]
        stable = subset["RELIABLE"]
        common = stable & np.isfinite(full) & np.isfinite(body)
        pair.append(dict(device=dev, original_reliable=int(stable.sum()), common_first_hits=int(common.sum()),
                         full_only=int((stable & np.isfinite(full) & ~np.isfinite(body)).sum()),
                         body_only=int((stable & np.isfinite(body) & ~np.isfinite(full)).sum()),
                         full_common_MAE_mm=errors(full, common, band)["MAE_mm"],
                         body_common_MAE_mm=errors(body, common, band)["MAE_mm"]))
    table(output / "KINECT8_FULL_BODY_COMMON_HITS.csv", pair)
    table(output / "KINECT8_FIRST_RANGE.csv", metrics)
    for first in (0, 4):
        panels(output / f"KINECT{first}-{first+3}_FIRST_RANGE.jpg", tuple(scenes), visuals[first:first+4],
               "Native first range   Green within   Red early   Blue late   Amber miss   Gray unreliable", (460, 274))
    summary = {}
    for name in scenes:
        row = next(r for r in metrics if r["candidate"] == name and r["device"] == "ALL" and r["region"] == "RELIABLE")
        ma = [r["MAE_mm"] for r in metrics if r["candidate"] == name and isinstance(r["device"], int)
              and r["region"] == "RELIABLE" and r["MAE_mm"] is not None]
        summary[name] = {**row, "equal_sensor_MAE_mm": float(np.mean(ma)) if ma else None,
                         "sensor_count_for_mean": len(ma)}
    return summary


def source_six(contract, scenes, output):
    with np.load(contract["original_inputs"]) as src, np.load(contract["original_targets"]) as target:
        K, E = (src[k][0] for k in ("K", "E"))
        masks = src["masks"][0].astype(bool)
        depth = target["depth"][0]
        valid = target["depth_valid"][0].astype(bool) & masks.astype(bool) & np.isfinite(depth) & (depth > 0)
    entries, first_body, face_body = [], [], []
    for i, cam in enumerate(SOURCE_IDS):
        for name, scene in scenes.items():
            first, face, _ = raster(scene, None, dict(K=K[i], E=E[i]), masks[i].shape)
            entries.append(dict(candidate=name, camera=cam, **errors(first - depth[i], valid[i], contract["kinect_band_m"]),
                                foreground_missing_kinect=int((masks[i] & ~valid[i]).sum()),
                                unknown_kinect_with_mesh_hit=int((masks[i] & ~valid[i] & (face >= 0)).sum())))
            if name == "BODY":
                first_body.append(first)
                face_body.append(face)
    table(output / "SIX_SOURCE_ORIGINAL_KINECT_CAMERA_Z.csv", entries)
    face = np.stack(face_body)
    body_parent = np.load(OUT / f"GW40_{contract['sample'].split('/')[1]}" / "human_body_only/ORIGINAL_FACE_INDICES.npy")
    parent_face = np.where(face >= 0, body_parent[np.maximum(face, 0)], -1)
    np.savez_compressed(output / "SIX_SOURCE_BODY_FIRST_SURFACE_518.npz", K=K, E=E, depth=np.stack(first_body),
                        face_id=face, parent_face_id=parent_face, valid=face >= 0, physical_camera_ids=SOURCE_IDS,
                        original_reference_mask=masks, original_Kinect_M=valid)


def orthographic(scene, mesh, bounds, degree, near):
    size = 390
    low, high = np.asarray(bounds)
    center = (low + high) / 2
    ang = np.deg2rad(degree)
    direction = np.array([np.sin(ang), 0, np.cos(ang)])
    right = np.array([np.cos(ang), 0, -np.sin(ang)])
    down = np.array([0, 1, 0])
    width = max(high[1] - low[1], np.abs(right) @ (high - low))
    grid = (np.arange(size) - (size - 1)/2) * (width * 1.08 / size)
    x, y = np.meshgrid(grid, grid)
    origin = center - direction * near + x[..., None] * right + y[..., None] * down
    return cast(scene, origin, np.broadcast_to(direction, origin.shape), mesh)[2]


def world_views(contract, meshes, scenes, output):
    metadata = {}
    for name, (v, f, c) in meshes.items():
        used = np.unique(f)
        metadata[name] = dict(point_count=len(v), referenced_points=len(used), faces=len(f),
                              full_world_bounds_m=[v.min(0).tolist(), v.max(0).tolist()],
                              used_world_bounds_m=[v[used].min(0).tolist(), v[used].max(0).tolist()])
        if name == "FULL":
            save_ply(output / "GW_FULL_TEXTURE999_VERTEX_POINTS.ply", v, c)
    whole = metadata["FULL"]["full_world_bounds_m"]
    near = float(np.max(np.abs(whole))) * 2 + 2
    for name, bounds in (("WORLD", whole), ("SUBJECT", contract["region_boxes_world_m"]["SUBJECT"])):
        groups = []
        for degree in (0, 90, 180, 270):
            images = [orthographic(scene, meshes[key], bounds, degree, near) for key, scene in scenes.items()]
            groups.append((f"{degree} deg", images))
        panels(output / f"{name}_SAME_WORLD_RGB_MESH.jpg", tuple(scenes), groups,
               f"{contract['sample']}   shared world metre scale", (405, 430))
    return metadata


def world_sections(contract, meshes, output):
    groups = []
    surfaces = {name: trimesh.Trimesh(vertices=mesh[0], faces=mesh[1], process=False)
                for name, mesh in meshes.items()}
    for name in GROUPS:
        low, high = np.asarray(contract["region_boxes_world_m"][name])
        for axis, idx in (("Y", 1), ("Z", 2)):
            dim = [i for i in range(3) if i != idx]
            mid = (low + high)/2
            span = (high[dim] - low[dim]).max() * 1.12
            size = 284
            images = []
            for surface in surfaces.values():
                normal = np.eye(3)[idx]
                seg = trimesh.intersections.mesh_plane(surface, normal, mid)
                coord = (seg[..., dim] - mid[dim]) / span
                trace = np.rint(coord * [size, -size] + size/2).astype(int)
                img = Image.new("RGB", (size, size), "#f8f9fc")
                pen = ImageDraw.Draw(img)
                pen.line([(0,size//2),(size,size//2)], fill="#dedfe1")
                pen.line([(size//2,0),(size//2,size)], fill="#dedfe1")
                for xy in trace[np.all((trace.min(1) <= size) & (trace.max(1) >= 0), axis=1)]:
                    pen.line([tuple(p) for p in xy], fill="#1c4c70", width=1)
                images.append(img)
            groups.append((f"{name} {axis}", images))
    for i in (0, 4, 8):
        panels(output / f"WORLD_SECTIONS_{i//4+1}.jpg", tuple(meshes), groups[i:i+4],
               "Coincident metric planes through fixed anatomical windows", (325, 306))


def run(case, freeze_only):
    contract, rows = input_contract(case)
    folder = OUT / f"GW40_{case}"
    qa = OUT / "GW40_TEACHER_QA" / case
    qa.mkdir(parents=True, exist_ok=True)
    frozen = qa / "FROZEN_TEACHER_QA.json"
    if frozen.exists() and read(frozen) != contract:
        raise ValueError("Frozen teacher evaluation evidence changed")
    if not frozen.exists():
        save(frozen, contract)
    if freeze_only:
        print(f"{case}: frozen TRAIN/HELD/BODY/native evaluation inputs", flush=True)
        return
    actual = read(folder / "FRAME_INPUT.json")
    if actual["sample"] != contract["sample"] or actual["train_ids"] != contract["train"] or actual["held_ids"] != contract["held"]:
        raise ValueError("Actual teacher FRAME_INPUT provenance differs from original camera roles")
    if {item["camera_id"] for item in actual["rows"]} != set(rows) or len(actual["rows"]) != len(rows):
        raise ValueError("Actual teacher FRAME_INPUT does not bind each of the 48 physical cameras")
    for item in actual["rows"]:
        ref = rows[item["camera_id"]]
        if any(item[k] != ref[k] for k in ("rgb", "mask", "K", "E")):
            raise ValueError("Actual teacher image/calibration binding differs")
    camera_path = folder / "GW_OFFICIAL_DIRECT_DEFAULT/cameras.json"
    bound = read(camera_path)
    camera_names = {f"cam_{cam:02d}" for cam in contract["train"]}
    if len(bound) != 40 or {Path(item["img_name"]).stem for item in bound} != camera_names:
        raise ValueError("Actual GaussianWrapping scene consumed cameras beyond the fixed TRAIN40")
    paths = {
        "FULL": folder / "GW_OFFICIAL_DIRECT_DEFAULT/mesh_ours_2pivots_post_texture_refined_999.ply",
        "BODY": folder / "human_body_only/GW_HUMAN_BODY_ONLY_DISPLAY.ply",
    }
    meshes = {k: read_ply(v) for k, v in paths.items()}
    points = folder / "human_body_only/GW_HUMAN_BODY_ONLY_POINTS.ply"
    proof = prove_body(rows, folder, meshes["FULL"], meshes["BODY"], points)
    proof.update(actual_GW_scene_camera_receipt=str(camera_path), actual_GW_scene_train_ids=contract["train"],
                 held_evaluation_ids=contract["held"])
    save(qa / "ORIGINAL_SUBSET_READBACK.json", proof)
    scenes = {k: mesh_scene(mesh) for k, mesh in meshes.items()}
    structure = world_views(contract, meshes, scenes, qa)
    measured = native(contract, scenes, qa)
    source_six(contract, scenes, qa)
    world_sections(contract, meshes, qa)
    optics = heldout(contract, rows, scenes, meshes, qa)
    with OLD_HELD.open(encoding="utf-8-sig", newline="") as f:
        historical = {r["candidate"]: float(r["micro_IoU"]) for r in csv.DictReader(f)}
    save(qa / "TEACHER_QA_SUMMARY.json", dict(
        sample=contract["sample"], status="PENDING_LOCAL_VISUAL_ADMISSION",
        meshes={k: str(v) for k, v in paths.items()}, body_points=str(points), structure=structure,
        subset=proof, HELD8=optics, native_Kinect8_reliable=measured,
        historical_0019=dict(held_body_micro_IoU=historical["GW_CLEAN"],
                             equal_sensor_first_range_MAE_mm=read(OLD_KINECT)["native_sensor_mae_mm"]),
    ))
    print(f"{case}: measured teacher evaluation complete, pending local visual review", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("0008", "0047"))
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    run(args.case, args.freeze_only)
