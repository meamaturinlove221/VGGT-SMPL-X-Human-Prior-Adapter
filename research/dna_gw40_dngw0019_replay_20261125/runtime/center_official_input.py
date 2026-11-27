import json
import math
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from PIL import Image

def center(root):
    contract = json.loads((root / 'KNOWN_POSE_SIFT_CONTRACT.json').read_text(encoding='utf-8-sig'))
    index = json.loads(Path(contract['source_index']).read_text(encoding='utf-8-sig'))
    rows = [r for r in index['rows'] if r['camera_id'] in index['train_ids']]
    # Expand the output canvas to contain the complete original pixel support, with
    # one interpolation-neighbour margin and a multiple-of-four canvas for r=2/4.
    width = 4 * math.ceil(max(2 * max(r['K'][0][2], r['rgb_header']['wh'][0]-1-r['K'][0][2]) + 3 for r in rows) / 4)
    height = 4 * math.ceil(max(2 * max(r['K'][1][2], r['rgb_header']['wh'][1]-1-r['K'][1][2]) + 3 for r in rows) / 4)
    dataset = root / 'official_input_train40'
    images = dataset / 'images'
    sparse = dataset / 'sparse' / '0'
    support = dataset / 'original_support_masks'
    for p in (images, sparse, support):
        p.mkdir(parents=True, exist_ok=True)
    recon = pycolmap.Reconstruction(str(root / 'known_pose_train40/sparse/0'))
    before = pycolmap.Reconstruction(str(root / 'known_pose_train40/sparse/0'))
    receipts = []
    max_ray = max_track = max_fx = max_world = 0.
    for row in rows:
        cid = row['camera_id'] + 1
        oldk = np.asarray(row['K'], dtype=np.float64)
        newk = oldk.copy()
        newk[0, 2], newk[1, 2] = (width-1)/2, (height-1)/2
        delta = newk[:2, 2] - oldk[:2, 2]
        h = np.array([[1., 0., delta[0]], [0., 1., delta[1]], [0., 0., 1.]])
        rgb = np.asarray(Image.open(row['rgb']).convert('RGB'))
        alpha = np.asarray(Image.open(row['mask']).convert('L'))
        rgba = np.concatenate((rgb, alpha[..., None]), axis=2)
        warped = cv2.warpAffine(rgba, h[:2], (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        Image.fromarray(warped, 'RGBA').save(images / f'cam_{row["camera_id"]:02d}.png')
        valid = cv2.warpAffine(np.full(alpha.shape, 255, dtype=np.uint8), h[:2], (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Image.fromarray(valid, 'L').save(support / f'cam_{row["camera_id"]:02d}.png')
        ys, xs = np.nonzero(alpha)
        bounds = [float(xs.min()+delta[0]), float(ys.min()+delta[1]), float(xs.max()+delta[0]), float(ys.max()+delta[1])]
        assert bounds[0] >= 0 and bounds[1] >= 0 and bounds[2] <= width-1 and bounds[3] <= height-1
        camera = recon.camera(cid)
        original_params = before.camera(cid).params.copy()
        camera.width, camera.height = width, height
        camera.params = np.array([oldk[0, 0], oldk[1, 1], width/2, height/2])
        image = recon.image(cid)
        for point in image.points2D:
            oldxy = point.xy.copy()
            point.xy = oldxy + delta
            # COLMAP track xy uses corner-origin pixels; both principal points +.5.
            ray_before = (oldxy - original_params[2:]) / original_params[:2]
            ray_after = (point.xy - camera.params[2:]) / camera.params[:2]
            max_ray = max(max_ray, float(np.max(np.abs(ray_after-ray_before))))
            max_track = max(max_track, float(np.max(np.abs(point.xy-oldxy-delta))))
        max_fx = max(max_fx, float(np.max(np.abs(camera.params[:2]-original_params[:2]))))
        receipts.append({'camera_id': row['camera_id'], 'image_name': image.name, 'H_OpenCV_pixel_centers': h.tolist(), 'original_K_OpenCV': oldk.tolist(), 'centered_K_OpenCV': newk.tolist(), 'centered_COLMAP_params': camera.params.tolist(), 'alpha_nonzero_bounds_transformed': bounds, 'source_rgb': row['rgb'], 'source_mask': row['mask'], 'track_coordinates_transformed': len(image.points2D)})
    for pid, point in recon.points3D.items():
        max_world = max(max_world, float(np.max(np.abs(point.xyz - before.point3D(pid).xyz))))
    recon.write(str(sparse))
    receipt = {'status': 'CENTERED_RGBA_COLMAP_INPUT_PREPARED_ZERO_TRAINING', 'images': len(rows), 'held_images': 0, 'canvas_width': width, 'canvas_height': height, 'resampling': 'cv2.warpAffine INTER_LINEAR; identical H for RGB/alpha, transparent black border storage; complete original support retained, no body crop', 'world_points': recon.num_points3D(), 'world_point_max_delta': max_world, 'focal_max_delta': max_fx, 'normalized_ray_max_error': max_ray, 'track_translation_max_error_pixels': max_track, 'opencv_center': '((width-1)/2,(height-1)/2)', 'colmap_center': '(width/2,height/2); existing +0.5 convention applied exactly once', 'GW_alpha_semantics': 'Official train composites alpha to default black background and uses full-image L1/SSIM; padded alpha0 is a transparent-background prior, not loss-ignored unknown data. No new loss mask.', 'new_SIFT': False, 'new_matching': False, 'new_triangulation': False, 'camera_pose_changed': False, 'rows': receipts}
    (root / 'CENTERED_INPUT_PROJECTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
