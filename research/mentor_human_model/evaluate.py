"""只评价已保存完整候选；扫描GT仅进入此离线评价，不进入infer。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sample_mesh(vertices,faces,count):
    """按全部三角形面积采样，不取最大组件或删碎片。"""
    rng = np.random.default_rng(20261016)
    triangles = vertices[faces]
    cross = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area = np.linalg.norm(cross,axis=-1)
    indices = rng.choice(len(faces),count,p=area/area.sum())
    uv = rng.random((count,2))
    root = np.sqrt(uv[:,0])
    weights = np.stack((1-root,root*(1-uv[:,1]),root*uv[:,1]),-1)
    return (triangles[indices]*weights[:,:,None]).sum(1),cross[indices]/area[indices,None]


def geometry_metrics(vertices,faces,scan_points,scan_normals,count,thresholds):
    """完整支持的双向表面采样距离，单位由用户明确给出。"""
    mesh_points,mesh_normals = sample_mesh(vertices,faces,count)
    target = scan_points.reshape(-1,3)
    normal = scan_normals.reshape(-1,3)
    prediction_distance,prediction_nearest = cKDTree(target).query(mesh_points,workers=4)
    target_distance,target_nearest = cKDTree(mesh_points).query(target,workers=4)
    scores = {}
    for threshold in thresholds:
        precision = float((prediction_distance<threshold).mean())
        recall = float((target_distance<threshold).mean())
        scores[str(threshold)] = dict(precision=precision,recall=recall,
            F_score=2*precision*recall/max(precision+recall,1e-12))
    return dict(mesh_sample_count=count,scan_sample_count=len(target),mesh_vertices=len(vertices),mesh_faces=len(faces),
        mesh_to_scan_mean=float(prediction_distance.mean()),scan_to_mesh_mean=float(target_distance.mean()),
        symmetric_chamfer_L1=.5*float(prediction_distance.mean()+target_distance.mean()),
        mesh_to_scan_p95=float(np.percentile(prediction_distance,95)),scan_to_mesh_p95=float(np.percentile(target_distance,95)),
        oriented_normal_cosine=.5*float((mesh_normals*normal[prediction_nearest]).sum(-1).mean()+
                                      (normal*mesh_normals[target_nearest]).sum(-1).mean()),
        absolute_normal_consistency=.5*float(np.abs((mesh_normals*normal[prediction_nearest]).sum(-1)).mean()+
                                            np.abs((normal*mesh_normals[target_nearest]).sum(-1)).mean()),thresholds=scores)


def main():
    """明确训练fit/开发角色，数值不自动宣布完整人体几何通过。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--role',choices=['training_fit','development','heldout_identity'],required=True)
    parser.add_argument('--length-unit',required=True)
    parser.add_argument('--sample-count',type=int,default=65536)
    parser.add_argument('--thresholds',type=float,nargs='+',default=[.005,.01,.02])
    args = parser.parse_args()
    with np.load(args.inference/'complete_mesh.npz',allow_pickle=False) as archive:
        vertices,faces = archive['vertices'],archive['faces']
    with np.load(args.targets,allow_pickle=False) as archive:
        target = {key:archive[key] for key in archive.files}
    with np.load(args.inference/'source_RAW.npz',allow_pickle=False) as archive:
        raw = {key:archive[key] for key in archive.files}
    valid = target['depth_valid'].astype(bool)
    normal_valid = target['normal_valid'].astype(bool) & raw['normal_valid'].astype(bool)
    difference = raw['depth'][valid]-target['depth'][valid]
    normal_cosine = (raw['world_normal']*target['world_normal']).sum(-1)[normal_valid]
    report = dict(role=args.role,length_unit=args.length_unit,
        source_depth=dict(valid_pixels=int(valid.sum()),L1=float(np.abs(difference).mean()),RMSE=float(np.sqrt((difference**2).mean()))),
        source_world_normal=dict(valid_pixels=int(normal_valid.sum()),mean_oriented_cosine=float(normal_cosine.mean())),
        geometry=geometry_metrics(vertices,faces,target['scan_points'],target['scan_normals'],args.sample_count,args.thresholds),
        NVS_photo_metrics='NOT_RUN: this evaluator received no real non-input query photographs',
        full_body_visual_geometry_pass='NOT_ADJUDICATED; numbers do not replace RAW/full-mesh/360 review',BODY_LOCKED=False,
        support='all mesh components, full area-sampled surface and every supplied legal scan point; no confidence filtering')
    path = args.inference/'EVALUATION.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
