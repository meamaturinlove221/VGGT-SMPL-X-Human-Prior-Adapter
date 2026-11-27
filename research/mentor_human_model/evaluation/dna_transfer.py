"""固定U2000迁移：完整mesh首表面与真实查询照片评价，不修改模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity


def export_mesh_first_surface(vertices, faces, cameras, out, h=518, w=518, device='cuda'):
    """复用已核验整数中心raster，保留所有三角形，以最近面产生mask/depth。"""
    import torch
    from research.mentor_human_model.model.existing_ops.p3d_bridge import prepare_dense_mesh_raster_cache_vectorized
    out = Path(out)
    vertices = torch.as_tensor(vertices, device=device, dtype=torch.float32)[None]
    faces = torch.as_tensor(faces, device=device, dtype=torch.long)
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            index = int(camera.get('query_index', index))
            height, width = int(camera.get('height', h)), int(camera.get('width', w))
            K = torch.as_tensor(camera['K'], device=device, dtype=torch.float32)[None, None]
            E = torch.as_tensor(camera['E'], device=device, dtype=torch.float32)[None, None]
            raster = prepare_dense_mesh_raster_cache_vectorized(vertices, faces, K, E[..., :3, :3],
                E[..., :3, 3], image_height=height, image_width=width, patch_size=1, face_chunk_size=32)
            mask = raster.silhouette[0, 0].cpu().numpy()
            depth = raster.visible_depth[0, 0].cpu().numpy()
            face_id = raster.visible_face_index[0, 0].cpu().numpy()
            directory = out / f'query_{index:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / 'mesh_first_surface.npz', mask=mask, depth=depth, face_id=face_id)
            Image.fromarray(mask.astype(np.uint8)*255).save(directory / 'mesh_mask.png')
            supported = depth[mask]
            if len(supported):
                lo, hi = supported.min(), supported.max()
                display = np.where(mask, (depth-lo)/max(float(hi-lo), 1e-8), 0)
                Image.fromarray((display.clip(0, 1)*255).astype(np.uint8)).save(directory / 'mesh_depth.png')
            print(json.dumps(dict(stage='complete_mesh_first_surface', query_index=index,
                                 vertices=vertices.shape[1], faces=len(faces))), flush=True)


def mask_metrics(prediction, truth):
    """全部像素上的轮廓交并比和双向边界像素距离，不把点占据当mesh。"""
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    intersection, union = np.logical_and(prediction, truth).sum(), np.logical_or(prediction, truth).sum()
    pred_boundary = prediction & ~binary_erosion(prediction)
    true_boundary = truth & ~binary_erosion(truth)
    result = dict(predicted_pixels=int(prediction.sum()), truth_pixels=int(truth.sum()),
                  intersection_pixels=int(intersection), union_pixels=int(union),
                  IoU=float(intersection/union) if union else None)
    if pred_boundary.any() and true_boundary.any():
        p = distance_transform_edt(~true_boundary)[pred_boundary]
        t = distance_transform_edt(~pred_boundary)[true_boundary]
        result.update(boundary_pred_to_truth_mean_px=float(p.mean()), boundary_truth_to_pred_mean_px=float(t.mean()),
                      boundary_symmetric_mean_px=float((p.mean()+t.mean())/2))
    else:
        result['boundary_distance'] = 'NOT_DEFINED: one boundary is empty; see pixel counts'
    return result


def photo_metrics(prediction, truth, mask):
    """RGB误差分母固定为真实前景或整图，SSIM中心同支持，不按预测alpha筛选。"""
    error = np.asarray(prediction, np.float64)-np.asarray(truth, np.float64)
    _, ssim_map = structural_similarity(truth, prediction, data_range=1., channel_axis=-1,
                                       full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    result = {}
    for name, support in [('full_frame', np.ones(mask.shape, bool)), ('real_foreground', mask)]:
        if not support.any():
            result[name] = 'NOT_RUN: no real foreground pixels'
            continue
        mse = float(np.square(error[support]).mean())
        result[name] = dict(pixels=int(support.sum()), scalar_channels=int(support.sum()*3),
                           L1=float(np.abs(error[support]).mean()), MSE=mse,
                           PSNR_dB=float(-10*np.log10(mse)) if mse else 'infinite (exact equality)',
                           SSIM=float(ssim_map[support].mean()))
    result['LPIPS'] = 'NOT_RUN: no verified local pretrained LPIPS evaluator configured'
    result['SSIM_definition'] = 'Gaussian sigma1.5, 11x11 window, population covariance; mean RGB SSIM map over fixed support centers (reflect border)'
    return result


def evaluate_photos(inference, photos_manifest):
    """只在模型产物保存之后读取独立照片清单；缺真值如实记录。"""
    inference, photos_manifest = Path(inference), Path(photos_manifest)
    manifest = json.loads(photos_manifest.read_text(encoding='utf-8'))
    rows = []
    for item in manifest['rows']:
        row = dict(query_index=item['query_index'], view_id=item['view_id'])
        rgb_path, mask_path = item.get('rgb_path'), item.get('mask_path')
        if not rgb_path or not mask_path:
            row['status'] = 'NOT_RUN: real query RGB or mask unavailable'
            rows.append(row)
            continue
        rgb_path, mask_path = Path(rgb_path), Path(mask_path)
        if not rgb_path.is_absolute():
            rgb_path = photos_manifest.parent/rgb_path
        if not mask_path.is_absolute():
            mask_path = photos_manifest.parent/mask_path
        truth = np.asarray(Image.open(rgb_path).convert('RGB'), np.float32)/255.
        mask = np.asarray(Image.open(mask_path).convert('L')) > 127
        query = inference/f"query_{item['query_index']:03d}"
        with np.load(query/'render.npz') as archive:
            prediction, alpha = archive['rgb'], archive['mask']
        if prediction.shape != truth.shape or mask.shape != truth.shape[:2]:
            raise ValueError('query photo/mask must use exactly the frozen query K image coordinates')
        row.update(status='RUN', rgb_path=str(rgb_path), mask_path=str(mask_path),
                   GS_photo=photo_metrics(prediction, truth, mask), GS_alpha=mask_metrics(alpha >= .5, mask),
                   GS_alpha_threshold=.5)
        mesh_path = query/'mesh_first_surface.npz'
        if mesh_path.exists():
            with np.load(mesh_path) as archive:
                row['complete_mesh_first_surface'] = mask_metrics(archive['mask'], mask)
        else:
            row['complete_mesh_first_surface'] = 'NOT_RUN: full triangle first-surface raster missing'
        delta = np.abs(prediction-truth).mean(-1)
        panels = [truth, prediction, np.repeat(delta[..., None], 3, -1),
                  np.stack((mask, alpha >= .5, np.zeros_like(mask)), -1).astype(float)]
        Image.fromarray((np.concatenate(panels, axis=1).clip(0, 1)*255).astype(np.uint8)).save(query/'PHOTO_GT_GS_ERROR_MASK.png')
        rows.append(row)
    report = dict(rows=rows, role=manifest.get('role', 'development'), BODY_LOCKED=False,
                  DNA_scan_Chamfer='NOT_RUN: no reliable independent clothed scan supplied',
                  Kinect_measurement='NOT_RUN: no independent Kinect evaluation support supplied',
                  support='real photo foreground and fixed full frame; no predicted-alpha denominator; triangle silhouette separate from GS alpha')
    (inference/'PHOTO_EVALUATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    """CPU离线照片评价入口，不训练、不读取checkpoint、不改归一化。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', required=True, type=Path)
    parser.add_argument('--photos', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_photos(args.inference, args.photos), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
