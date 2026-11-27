"""Export the archived DN_GW U200 six-source RAW as 2M-point RGB display copies."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

from interpolate_raw_dn0019 import (
    COLORED, COUNT, ROOT, SEED, eligible_cells, fill, preview, read,
    read_vertices, save_ply, silhouette_votes,
)

from research.dna_b_adapter_20261021.teacher_data import inverse_rt


EVAL = ROOT / 'evaluation/DN_GW_U200'
DEST = ROOT.parent / 'DNA_DN_GW_U200_SIX_VIEW_BODY_DISPLAY_2M_20261125'
SUPPORT_IDS = (2, 26, 11, 35)
FIRST = ('DNA_RENDERING__0034__0034_04__0224', 'DNA_RENDERING__0097__0097_04__0000')
REF = ROOT / 'display_dense/RAW_BODY_DENSE_RGB/RECEIPT.json'
INTERPOLATION = ('All four adjacent pixels are retained finite positive-depth points from the same source. '
                 'Camera-Z span and all four world edges <= 2 * minimum corner Z / min(fx,fy). '
                 'Bilinear camera-Z and RAW RGB, then calibrated subpixel ray unprojection.')


def save_json(path, payload):
    import json

    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def load_source(case):
    source = case / 'GIVEN_PRIOR'
    inference = read(source / 'INFERENCE.json')
    inputs = Path(inference['input_npz'])
    lineage = read(inputs.with_name('LINEAGE.json'))
    if (inference['id'] != case.name.replace('__', '/') or inference['id'] != lineage['sample_id']
            or Path(inference['checkpoint']) != ROOT / 'student/DN_GW/checkpoint_u0200.pt'
            or inference['prior_role'] != 'GIVEN_PRIOR' or Path(inference['output']) != source):
        raise ValueError(f'Wrong branch, source or checkpoint: {case.name}')
    with np.load(source / 'source_RAW.npz') as raw_archive, np.load(inputs) as batch:
        data = {key: raw_archive[key][0] for key in ('world_points', 'mask', 'depth', 'K', 'E')}
        rgb = np.clip(batch['images'][0].transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
        if not all(np.array_equal(data[key], batch[input_key][0])
                   for key, input_key in (('mask', 'masks'), ('K', 'K'), ('E', 'E'))):
            raise ValueError(f'Archived masks or calibration differ from frozen inputs: {case.name}')
    data['mask'] = data['mask'].astype(bool, copy=False)
    fg = data['mask']
    raw = read_vertices(source / 'source_RAW_all_foreground.ply', COLORED)
    if (fg.shape != (6, 518, 518) or len(raw) != inference['metrics']['raw_foreground_count']
            or not np.array_equal(raw['xyz'], data['world_points'][fg])
            or not np.array_equal(raw['rgb'], rgb[fg])):
        raise ValueError(f'RAW PLY / original mask, world XYZ or RGB mismatch: {case.name}')
    index = np.full(fg.shape, -1, dtype=np.int32)
    index[fg] = np.arange(len(raw), dtype=np.int32)
    return source, inference, lineage, raw, data, rgb, index


def frame_support(inference, lineage, data):
    native_name, group = lineage['source_lineage']['calibration_source'].split('::')
    native = Path(native_name)
    frame = int(inference['id'].split('/')[-1])
    with native.open('rb') as stream, h5py.File(stream, 'r') as ann:
        calibration = ann[group]
        cameras = sorted(int(key) for key in calibration)
        extrinsics = np.stack([inverse_rt(calibration[f'{key:02}']['RT'][()]) for key in cameras])
        error = np.max(np.abs(data['E'][:, None] - extrinsics[None]), axis=(2, 3))
        assignment = error.argmin(axis=1)
        ids = [cameras[slot] for slot in assignment]
        closest = np.sort(error, axis=1)
        if len(set(ids)) != 6 or np.any(closest[:, 0] > 1e-5) or np.any(closest[:, 1] < 1e-3):
            raise ValueError(f'Cannot bind each RAW slot to unique native extrinsics: {inference["id"]}: {closest[:, :2]}')
        slot_rows = [dict(slot=slot, camera_id=cam, archival_lineage_id=int(lineage['source_view_ids'][slot]),
                          original_calibration=f'{native}::{group}/{cam:02}/RT',
                          max_abs_extrinsic_residual=float(closest[slot, 0]))
                     for slot, cam in enumerate(ids)]
        masks = []
        sources = []
        for cam in SUPPORT_IDS:
            mask_key = f'Mask/{cam}/mask/{frame}'
            mask = cv2.imdecode(ann[mask_key][()], cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask.max(2)
            h, w = mask.shape
            cal = calibration[f'{cam:02}']
            k, distort = cal['K'][()], cal['D'][()]
            e = extrinsics[cameras.index(cam)]
            mx, my = cv2.initUndistortRectifyMap(k, distort, None, k, (w, h), cv2.CV_32FC1)
            undistorted = cv2.remap(mask, mx, my, cv2.INTER_NEAREST)
            clean = cv2.resize(undistorted, (1224, 1024), interpolation=cv2.INTER_NEAREST) > 0
            rim = cv2.dilate(clean.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
            sx, sy = 1224 / w, 1024 / h
            a = np.array([[sx, 0, .5 * (sx - 1)], [0, sy, .5 * (sy - 1)], [0, 0, 1]])
            scaled_k = np.einsum('ij,jk->ik', a, k, optimize=False)
            masks.append((cam, scaled_k, e, rim))
            sources.append(dict(camera_id=cam, mask=f'{native}::{mask_key}',
                                calibration=f'{native}::{group}/{cam:02}', native_hw=[h, w],
                                output_hw=[1024, 1224], K=scaled_k.tolist(), E=e.tolist(),
                                foreground_pixels=int(clean.sum())))
    return slot_rows, masks, sources


def audit_case(case):
    source, inference, lineage, raw, data, rgb, index = load_source(case)
    slots, masks, support = frame_support(inference, lineage, data)
    folder = DEST / case.name
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    sheet = Image.new('RGB', (1554, 1090), 'white')
    for slot, camera in enumerate(slots):
        fg = data['mask'][slot]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        main = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        img = Image.fromarray(rgb[slot])
        draw = ImageDraw.Draw(img)
        for component in range(1, count):
            if component == main:
                continue
            rect = stats[component]
            x, y, w, h, points = map(int, rect)
            region = labels[y:y + h, x:x + w] == component
            raw_index = index[slot, y:y + h, x:x + w][region]
            votes, by_camera = silhouette_votes(raw['xyz'][raw_index], masks)
            protected = int((votes >= 3).sum())
            records.append(dict(slot=slot, camera=camera['camera_id'], original_mask_component=component,
                                pixel_bounds_xyxy=[x, y, x+w-1, y+h-1], points=points,
                                silhouette_support_per_camera=by_camera,
                                points_with_at_least_three_votes=protected))
            if not protected:
                draw.rectangle((x-1, y-1, x+w, y+h), outline='#ff8a00', width=1)
                if points >= 4:
                    draw.text((x, max(0, y-10)), str(component), fill='#ff8a00', stroke_width=1,
                              stroke_fill='#101820')
        left, top = (slot % 3) * 518, (slot // 3) * 545
        ImageDraw.Draw(sheet).text((left+5, top+5),
                                   f'Frozen RGB | slot {slot} | physical camera {camera["camera_id"]:02d}',
                                   fill='black')
        sheet.paste(img, (left, top+25))
    figure = folder / 'ORIGINAL_SOURCE_MASK_ENVIRONMENT_AUDIT.png'
    sheet.save(figure)
    audit = dict(sample=inference['id'], source_slots=slots, silhouette_support_sources=support,
                 components=records, figure=str(figure),
                 figure_legend='Orange outlines are non-main original mask components with no point reaching 3 of 4 silhouette votes.')
    save_json(folder / 'COMPONENT_REVIEW.json', audit)
    print(f'AUDITED {case.name} {[(r["slot"],r["camera"],r["original_mask_component"],r["points"]) for r in records if not r["points_with_at_least_three_votes"] and r["points"] >= 12]}', flush=True)


def export_case(case, decisions):
    source, inference, lineage, raw, data, rgb, index = load_source(case)
    folder = DEST / case.name
    audit = read(folder / 'COMPONENT_REVIEW.json')
    output = folder / f'DN_GW_U200_{case.name[15:]}_RAW_INTERPOLATED_BODY_DENSE_RGB_2000000_POINTS.ply'
    receipt = folder / 'RECEIPT.json'
    if receipt.is_file() and output.is_file():
        print(f'EXISTING {case.name} {output}', flush=True)
        return
    keep = data['mask'].copy()
    candidates = {(r['slot'], r['original_mask_component']): r for r in audit['components']}
    removed_components = []
    for selection in decisions['components']:
        slot, component = selection['slot'], selection['component']
        row = candidates[(slot, component)]
        if row['points_with_at_least_three_votes']:
            raise ValueError(f'Body support protects component {case.name} {slot}/{component}')
        labels = cv2.connectedComponentsWithStats(data['mask'][slot].astype(np.uint8), connectivity=8)[1]
        region = labels == component
        if region.sum() != row['points']:
            raise ValueError(f'Component changed since audit: {case.name} {slot}/{component}')
        keep[slot, region] = False
        removed_components.append(dict(row, reason=selection['reason']))
    cells, parents, before_continuity = eligible_cells(data, index, keep)
    kept = index[keep]
    removed = index[data['mask'] & ~keep]
    n_new = COUNT - len(kept)
    rng = np.random.default_rng(SEED)
    if not len(cells) or n_new < 1:
        raise ValueError(f'No eligible native-cell interpolation: {case.name}')
    counts = np.full(len(cells), n_new // len(cells), dtype=np.int32)
    counts[rng.permutation(len(cells))[:n_new % len(cells)]] += 1
    uv = rng.integers(0, 65536, size=(n_new, 2), dtype=np.uint16)
    result = np.empty(COUNT, dtype=COLORED)
    result[:len(kept)] = raw[kept]
    fill(result[len(kept):], raw, data, cells, parents, counts, uv)
    save_ply(output, result)
    camera_ids = [row['camera_id'] for row in audit['source_slots']]
    provenance = folder / 'POINT_PROVENANCE.npz'
    np.savez_compressed(provenance, camera_ids=np.asarray(camera_ids), original_raw_index=kept,
                        original_view_y_x=np.argwhere(keep).astype(np.int16), removed_original_raw_index=removed,
                        removed_view_y_x=np.argwhere(data['mask'] & ~keep).astype(np.int16),
                        cell_source_y_x=cells, cell_parent_raw_indices=parents, cell_generated_counts=counts,
                        generated_uv_q16=uv)
    figure = folder / 'RAW_BODY_DENSE_RGB_PREVIEW.png'
    preview(figure, raw, result)
    by_source = np.bincount(cells[:, 0], weights=counts, minlength=6).astype(int).tolist()
    report = dict(status='RAW_BODY_DENSE_RGB_COMPLETE', sample_id=inference['id'], output=str(output),
                  output_bytes=output.stat().st_size, checkpoint=inference['checkpoint'],
                  inference_receipt=str(source / 'INFERENCE.json'), input_npz=inference['input_npz'],
                  input_lineage=str(Path(inference['input_npz']).with_name('LINEAGE.json')),
                  source_camera_ids=camera_ids, source_slots=audit['source_slots'],
                  source_RAW_PLY=str(source / 'source_RAW_all_foreground.ply'),
                  source_RAW_archive=str(source / 'source_RAW.npz'), reference_receipt=str(REF),
                  original_RAW_points=len(raw), removed_original=len(removed), retained_original=len(kept),
                  new_interpolated=n_new, points=COUNT, seed=SEED,
                  original_foreground_per_source=data['mask'].sum((1, 2)).tolist(),
                  retained_original_per_source=keep.sum((1, 2)).tolist(), generated_per_source=by_source,
                  original_points_and_RGB_exact=bool(np.array_equal(result[:len(kept)], raw[kept])),
                  readback_xyz_and_rgb_exact=True,
                  cleaning=dict(audit=audit['figure'], component_review=str(folder / 'COMPONENT_REVIEW.json'),
                                silhouette_support_sources=audit['silhouette_support_sources'],
                                rule='Only visually identified detached floor/light components lacking a point with >=3 of 4 same-frame native silhouette votes with the original 8px rim are removed. Ambiguous or attached components remain.',
                                visual_review=decisions['review'], removed_components=removed_components,
                                original_mask_modified=False),
                  interpolation=dict(candidate_cells=before_continuity, accepted_cells=len(cells),
                                     rejected_discontinuous_cells=before_continuity-len(cells), rule=INTERPOLATION),
                  provenance=str(provenance), provenance_rule='First retained_original rows follow original_raw_index. Remaining rows group by cell order and cell_generated_counts; each uv=(generated_uv_q16+.5)/65536, parents TL/TR/BL/BR.',
                  preview=str(figure), scope='Display interpolation of existing RAW. Residual silhouette and multiview geometry errors may remain.',
                  network=0, mesh=0, DPSR=0, refiner=0, GPU=0, optimizer=0)
    save_json(receipt, report)
    print(f'EXPORTED {case.name} {len(kept)}+{n_new}={COUNT}; removed {len(removed)}; {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'export'))
    parser.add_argument('cases', nargs='*')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cases = args.cases or [*FIRST, *sorted(p.name for p in EVAL.glob('DNA_RENDERING*')
                                          if p.name not in FIRST and '__0019__' not in p.name)]
    decisions = read(DEST / 'DISPLAY_COMPONENT_SELECTION.json') if args.action == 'export' else None
    for case in cases:
        if args.action == 'audit':
            audit_case(EVAL / case)
        else:
            export_case(EVAL / case, decisions[case])


if __name__ == '__main__':
    main()
