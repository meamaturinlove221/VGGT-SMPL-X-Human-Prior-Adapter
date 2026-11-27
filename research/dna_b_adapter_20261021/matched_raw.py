"""零更新恢复三模型完整RAW，只在实际同输入同支持时计算米制误差。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from matched_raw_figures import make_figures
from source_data import source_metadata

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/thuman20/DNA_TEACHER_COMPARE_20261023'
OLD=ROOT/'results/DNA_LEGACY_RESTORE_20261019/COLORED_RAW_COMPARE'
B=ROOT/'artifacts/thuman20/DNA_INTEGRATED_REPAIR_20261021'


def read_json(path):
    """只读已有来源收据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    """仅向本轮结果根写入。"""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def rgb8(images):
    """按既有源颜色规则转为PLY八位RGB，不改变几何。"""
    return np.uint8(np.clip(np.rint(images*255 if images.dtype.kind=='f' else images),0,255))


def write_ply(path,points,colors,views):
    """写所有原前景点和源索引，不应用置信度或测量有效性过滤。"""
    data=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                    ('red','u1'),('green','u1'),('blue','u1'),('view_id','u1')])
    for i,key in enumerate(['x','y','z']):data[key]=points[:,i]
    for i,key in enumerate(['red','green','blue']):data[key]=colors[:,i]
    data['view_id']=views
    header=('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property uchar view_id\nelement face 0\n'
            'property list uchar int vertex_indices\nend_header\n')%len(points)
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'));data.tofile(stream)


def load_models(slug,b_folders=None):
    """绑定10月21日B缓存，避免误用10月17日另一次恢复输入。"""
    gdir=OLD/slug/'OLD_G2_U300';identity=read_json(gdir/'SOURCE_IDENTITY.json')
    with np.load(gdir/'RESTORED_RAW.npz') as z:
        g={key:z[key] for key in ['world_points','depth_physical','images','mask','K','E']}
    with np.load(identity['source_prediction']) as z:
        g['rgb_float']=z['RGB'].transpose(0,2,3,1)
    g.update(name='OLD_G2_U300',raw_path=str(gdir/'RESTORED_RAW.npz'),
             checkpoint=identity['checkpoint'],input_path=identity['input_source'],
             source_scale=identity['source_scale'],source_identity=str(gdir/'SOURCE_IDENTITY.json'))
    inputs=B/'inputs'/slug/'GIVEN_PRIOR/INPUTS.npz'
    with np.load(inputs) as z:
        rgb=z['images'][0].transpose(0,2,3,1);prior=z['smpl_vertices'][0]
    records=[g]
    for name,sub in [('B_U2000','B0'),('B_U600','B_KINECT_U600')]:
        folder=Path(b_folders[name]) if b_folders else B/'evaluation'/sub/slug/'GIVEN_PRIOR'
        info=read_json(folder/'INFERENCE.json')
        with np.load(folder/'source_RAW.npz') as z:
            r={key:z[key][0] for key in ['world_points','mask','K','E']}
            r.update(depth_physical=z['depth'][0],source_scale=float(z['source_camera_scale'][0]))
        r['E']=r['E'][:,:3]
        r.update(images=rgb8(rgb),rgb_float=rgb,name=name,raw_path=str(folder/'source_RAW.npz'),
                 checkpoint=info['checkpoint'],input_path=str(inputs),source_identity=str(folder/'INFERENCE.json'))
        records.append(r)
    return records,prior


def same_inputs(records):
    """实际数组相等才解锁共同支持指标；历史文字不能覆盖失败。"""
    reference=records[0];checks=[]
    for record in records[1:]:
        values={key:bool(np.array_equal(reference[key],record[key]))
                for key in ['rgb_float','mask','K','E']}
        checks.append(dict(model=record['name'],equal_to='OLD_G2_U300',**values))
    return dict(comparable=all(all(r[k] for k in ['rgb_float','mask','K','E']) for r in checks),
                checks=checks,definition='Exact source float RGB, foreground mask, pixel K and physical E arrays.')


def metric_rows(records,targets,matched):
    """同像素camera-Z与位置误差；缺测区域保持未知。"""
    if not matched:
        return [dict(model=r['name'],status='NOT_COMPARABLE',reason='actual source arrays differ') for r in records]
    with np.load(targets) as z:
        depth=z['depth'][0];valid=z['depth_valid'][0].astype(bool)
    rows=[]
    for record in records:
        mask=record['mask'].astype(bool)
        support=valid&mask&np.isfinite(depth)&(depth>0)
        K,E=record['K'],record['E']
        yy,xx=np.indices(depth.shape[-2:],dtype=np.float32)
        # 原K无偏斜；显式相机射线避免本机BLAS矩阵逆异常。
        if np.any(K[:,0,1]!=0) or np.any(K[:,1,0]!=0):
            raise ValueError('Observed skewed K requires its real inverse; do not use zero-skew ray formula.')
        rays=np.stack([(xx[None]-K[:,0,2,None,None])/K[:,0,0,None,None],
                       (yy[None]-K[:,1,2,None,None])/K[:,1,1,None,None],
                       np.ones_like(depth)],axis=-1)
        camera=rays*depth[...,None]
        target_world=np.einsum('vhwi,vij->vhwj',camera-E[:,None,None,:3,3],E[:,:,:3],optimize=False)
        dz=np.abs(record['depth_physical']-depth)[support]
        distance=np.linalg.norm(record['world_points']-target_world,axis=-1)[support]
        finite=bool(np.isfinite(dz).all() and np.isfinite(distance).all())
        rows.append(dict(model=record['name'],status='MEASURED' if finite else 'NONFINITE_PREDICTION',
                         units='metres',support_pixels=int(support.sum()),foreground_pixels=int(mask.sum()),
                         missing_measurement_foreground=int((mask&~support).sum()),
                         camera_Z_abs_mean=float(dz.mean()),camera_Z_abs_P95=float(np.quantile(dz,.95)),
                         position_mean=float(distance.mean()),position_P95=float(np.quantile(distance,.95)),
                         scope='same original calibrated source pixels; Kinect valid region only'))
    return rows


def write_csv(path,rows):
    """按实际字段写短表，不为缺失指标制造数值。"""
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(sample):
    """先交完整源点，再记录可比范围；没有网络或优化器调用。"""
    slug=sample.replace('/','__');out=OUT/slug;out.mkdir(parents=True,exist_ok=True)
    records,prior=load_models(slug);contract=same_inputs(records)
    input_lineage=B/'inputs'/slug/'GIVEN_PRIOR/LINEAGE.json'
    actual_ids,actual_lineage=source_metadata(read_json(input_lineage))
    inputs=[]
    for record in records:
        folder=out/record['name'];folder.mkdir(exist_ok=True)
        mask=record['mask'].astype(bool);views=np.broadcast_to(np.arange(len(mask),dtype=np.uint8)[:,None,None],mask.shape)
        record.update(points=record['world_points'][mask],colors=record['images'][mask],view_id=views[mask])
        archive={key:record[key] for key in ['world_points','depth_physical','images','mask','K','E']}
        np.savez_compressed(folder/'COMPLETE_SOURCE_RAW.npz',**archive)
        write_ply(folder/'ALL_FOREGROUND_RGB_RAW.ply',record['points'],record['colors'],record['view_id'])
        for view in range(len(mask)):
            selected=record['view_id']==view
            write_ply(folder/f'SOURCE_{view:02d}_RGB_RAW.ply',record['points'][selected],record['colors'][selected],record['view_id'][selected])
        item={key:record[key] for key in ['name','checkpoint','raw_path','input_path','source_identity','source_scale']}
        item.update(sample=sample,foreground_points=int(mask.sum()),units='original calibrated metres',
                    source_camera_ids=','.join(actual_ids),
                    source_camera_identity_evidence=str(input_lineage)+'::source_metadata actual_identity_evidence',
                    camera_role='REFERENCE_INPUT_CALIBRATION',prior_role='GIVEN_PRIOR',
                    representation='complete foreground depth-backprojection RAW; no mesh or Gaussian claim',
                    all_foreground_ply=str(folder/'ALL_FOREGROUND_RGB_RAW.ply'),complete_npz=str(folder/'COMPLETE_SOURCE_RAW.npz'),
                    nonfinite_foreground=int((~np.isfinite(record['points']).all(1)).sum()),
                    checkpoint_binding='Recovered output provenance; no new model load in this CPU export')
        write_json(folder/'SOURCE_IDENTITY.json',item);inputs.append(item)
    rows=metric_rows(records,B/'inputs'/slug/'GIVEN_PRIOR/TARGETS.npz',contract['comparable'])
    figures=make_figures(records,out,prior)
    write_json(out/'MATCHED_INPUT_ARRAYS.json',contract)
    write_json(out/'FIGURE_INDEX.json',figures)
    write_csv(OUT/'MODEL_COMPARE_INPUTS.csv',inputs)
    write_csv(OUT/'RAW_MATCHED_METRICS.csv',rows)
    write_json(out/'EXECUTION_RECEIPT.json',dict(new_network_forwards=0,new_optimizer_updates=0,new_DPSR_calls=0,
               actual_models=[r['name'] for r in records],input_arrays=contract,metrics=rows,
               cross_view_correspondence='NOT_RUN: no independently frozen same-surface correspondence support supplied; NN is not substituted',
               heldout_silhouette='NOT_RUN: this restored packet has six source views only; self-view RAW occupancy is not independent silhouette accuracy'))
    print(json.dumps(dict(output=str(out),input_arrays=contract,metrics=rows),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample',default='DNA_RENDERING/0034/0034_04/0224')
    run(parser.parse_args().sample)
