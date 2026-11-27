"""只读0019现存数组与面；CPU统计和同窗展示，不导入模型。"""
from pathlib import Path
import csv, json, math
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from research.dna_gw40_dngw0019_replay_20261125.paths import QA_FONT, REPO as ROOT
OUT = ROOT / 'artifacts/thuman20/DNA_0019_LOCAL_CPU_REVIEW_20261023'
AUX = Path(__file__).resolve().parent
PHASES = ['U600', 'BK', 'BT', 'BTF']
FONT = ImageFont.truetype(str(QA_FONT), 18)


def load(path):
    """读取现存文件，保留原数组值。"""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stats(values):
    """空支持保持缺失，不补成零误差。"""
    x = np.asarray(values)
    return {'n': int(x.size), 'mean': float(x.mean()) if x.size else None,
            'p95': float(np.quantile(x, .95)) if x.size else None}


def angle(a, b):
    """向量角差只用于已有法线差异，绝不称真值误差。"""
    aa = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    bb = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.rad2deg(np.arccos(np.clip((aa * bb).sum(-1), -1, 1)))


def world(depth, k, e):
    """按原整数像素、camera-Z和world-to-camera语义反投影目标。"""
    yy, xx = np.indices(depth.shape[-2:], dtype=depth.dtype)
    camera = np.stack(((xx-k[:, 0, 2, None, None])*depth/k[:, 0, 0, None, None],
                       (yy-k[:, 1, 2, None, None])*depth/k[:, 1, 1, None, None], depth), -1)
    return np.einsum('shwj,sjk->shwk', camera-e[:, None, None, :3, 3], e[:, :3, :3])


def project(points, azimuth):
    """仅显示相机旋转，DNA负Y向上。"""
    t = math.radians(azimuth)
    return np.column_stack((points[:, 0]*math.cos(t)+points[:, 2]*math.sin(t),
                            -points[:, 1], -points[:, 0]*math.sin(t)+points[:, 2]*math.cos(t)))


def inside(points, box):
    """冻结世界盒，仅描述固定空间支持而非解剖真值。"""
    return ((points>=np.array(box[0]))&(points<=np.array(box[1]))).all(-1)


def crossing(field):
    """只计已存场的八角跨零cell，不提取新mesh。"""
    corners=[field[x:x+127,y:y+127,z:z+127] for x in [0,1] for y in [0,1] for z in [0,1]]
    low=np.minimum.reduce(corners);high=np.maximum.reduce(corners)
    return (low<=0)&(high>=0)&(low<high)


def panel(points, colors, faces, angle_deg, box, size):
    """所有点或所有三角面均参与；共同窗口之外仅显示裁切。"""
    w, h = size
    p = project(points, angle_deg)
    l, r, b, t = box
    factor = min((w-12)/(r-l), (h-24)/(t-b))
    uv = np.column_stack(((p[:, 0]-(l+r)/2)*factor+w/2,
                           ((b+t)/2-p[:, 1])*factor+(h+16)/2))
    im = Image.new('RGB', size, (248,248,248))
    draw = ImageDraw.Draw(im)
    if faces is None:
        ij = np.rint(uv).astype(int)
        good = np.isfinite(p).all(-1)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=18)&(ij[:,1]<h)
        indices = np.flatnonzero(good)
        indices = indices[np.argsort(p[indices,2], kind='stable')]
        canvas = np.array(im)
        canvas[ij[indices,1],ij[indices,0]] = colors[indices]
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)
    else:
        triangles = p[faces]
        normal = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True),1e-10)
        shade = (.35+.6*np.abs(normal[:,2]))*220
        for i in np.argsort(p[faces,2].mean(1)):
            q=uv[faces[i]]
            if q[:,0].max()<0 or q[:,0].min()>=w or q[:,1].max()<18 or q[:,1].min()>=h:
                continue
            c=int(shade[i]);draw.polygon([tuple(v) for v in q], fill=(c,c,c))
    draw.rectangle((0,0,w,20), fill='white')
    draw.text((5,1), f'{angle_deg} deg',font=FONT,fill='black')
    return im


def main():
    """固定四组同数据支持，一次统计并生成唯一交付对照图。"""
    index=json.loads((ROOT/'artifacts/thuman20/DNA_TEACHER_COMPARE_20261023/REVIEW_ARRAYS_INDEX.json').read_text())
    records={r['phase']:r for r in index if r['sample_id']=='DNA_RENDERING/0019/0019_06/0224'}
    reviews={p:load(records[p]['review_arrays']) for p in PHASES}
    inp=load(records['U600']['input_rgb']);target=load(records['U600']['target'])
    prior=inp['smpl_vertices'][0]
    config=json.loads((AUX/'0019_roi_config.json').read_text(encoding='utf-8-sig'))
    low, high=prior.min(0),prior.max(0);height=float(high[1]-low[1])
    bands={name:(float(low[1]+start*height),float(low[1]+end*height))
           for name,(start,end) in config['bands'].items()}
    arm_box=([float(low[0]-.1),bands['ARMS'][0],float(low[2]-.1)],
             [float(high[0]+.1),bands['ARMS'][1],float(low[2]+.35*(high[2]-low[2]))])
    ref=reviews['U600'];target_world=world(target['depth'][0],ref['K'][0],ref['E'][0])
    binding={'K':np.array_equal(inp['K'],ref['K']),'E':np.array_equal(inp['E'],ref['E']),
             'mask':np.array_equal(inp['masks'],ref['mask']),
             'target_depth':np.array_equal(target['depth'],ref['target_depth']),
             'target_valid':np.array_equal(target['depth_valid'],ref['target_depth_valid'])}
    if not all(binding.values()):raise ValueError('Original input/target differs from indexed REVIEW: '+str(binding))
    allrows=[];summary={'units':'source calibrated world units; no fitted alignment',
                        'bands_world_y':bands,'distal_arm_world_box':arm_box,'original_input_target_binding':binding,
                        'config':config,'models':{},'paths':records}
    meshes={};rawpoints={};samplepoints={};colors={}
    for phase in PHASES:
        a=reviews[phase];raw=load(records[phase]['original_complete_RAW'])
        fg=a['mask'][0];valid=fg&target['depth_valid'][0]
        nv=a['normal_valid'][0].astype(bool);p=a['world_points'][0]
        rawpoints[phase]=p[fg];samplepoints[phase]=raw['oriented_points'][0]
        colors[phase]=np.uint8(np.clip(a['RGB'][0].transpose(0,2,3,1)[fg],0,1)*255)
        d=np.abs(a['depth'][0]-target['depth'][0]);direct=np.linalg.norm(p-target_world,axis=-1)
        learned=a['world_normal'][0];geom=raw['geometric_normal'][0];refn=ref['world_normal'][0]
        goodlearn=np.isfinite(learned).all(-1)&(np.linalg.norm(learned,axis=-1)>1e-6)
        goodgeom=np.isfinite(geom).all(-1)&(np.linalg.norm(geom,axis=-1)>1e-6)
        goodref=np.isfinite(refn).all(-1)&(np.linalg.norm(refn,axis=-1)>1e-6)
        normal=angle(learned,geom);goodnormal=nv&goodlearn&goodgeom
        delta=angle(learned,refn);common=nv&ref['normal_valid'][0].astype(bool)&goodlearn&goodref
        stride=nv[:,::4,::4]
        rebuilt=p[:,::4,::4][stride]
        rebuilt_n=a['world_normal'][0,:,::4,::4][stride]
        sample_exact=np.array_equal(rebuilt,raw['oriented_points'][0])
        normal_exact=np.array_equal(rebuilt_n,raw['oriented_normals'][0])
        model={'foreground_n':int(fg.sum()),'measurement_n':int(valid.sum()),'missing_n':int((fg&~valid).sum()),
               'depth':stats(d[valid]),'direct':stats(direct[valid]),'normalvalid_n':int(nv.sum()),
               'stride4_n':int(stride.sum()),'saved_points_exact_stride4':sample_exact,
               'saved_normals_exact_stride4':normal_exact,'normal_vs_geom_deg':stats(normal[goodnormal]),
               'normal_vs_U600_deg':stats(delta[common]),'local':{}}
        target_arm=valid&inside(target_world,arm_box);raw_arm=fg&inside(p,arm_box)
        normal_arm=nv&inside(p,arm_box);sample_arm=stride&inside(p[:,::4,::4],arm_box)
        model['distal_arm']={'depth':stats(d[target_arm]),'direct':stats(direct[target_arm]),
                             'RAW_n':int(raw_arm.sum()),'normalvalid_n':int(normal_arm.sum()),
                             'saved_stride4_n':int(sample_arm.sum())}
        for view in range(6):
            row={'model':phase,'view_index':view,'units':'depth/Direct=source calibrated world units; normal=degrees; field=indicator units','foreground_n':int(fg[view].sum()),
                 'measurement_n':int(valid[view].sum()),'missing_measurement_n':int((fg[view]&~valid[view]).sum()),
                 'normalvalid_n':int(nv[view].sum()),'normalvalid_fraction_FG':float(nv[view].sum()/fg[view].sum()),
                 'stride4_n':int(stride[view].sum()),'saved_points_exact_stride4':sample_exact,
                 'saved_normals_exact_stride4':normal_exact,
                 'geometric_bad_in_normalvalid_n':int((nv[view]&~goodgeom[view]).sum()),
                 'learned_bad_in_normalvalid_n':int((nv[view]&~goodlearn[view]).sum()),
                 'normal_vs_geometric_excluded_n':int((nv[view]&~goodnormal[view]).sum()),
                 'normal_vs_U600_excluded_n':int((nv[view]&ref['normal_valid'][0,view].astype(bool)&~common[view]).sum()),
                 'distal_arm_RAW_n':int(raw_arm[view].sum()),'distal_arm_normalvalid_n':int(normal_arm[view].sum()),
                 'distal_arm_saved_stride4_n':int(sample_arm[view].sum())}
            for label,values,support in [('depth_abs',d,valid),('Direct',direct,valid),
                                          ('normal_vs_geometric_deg',normal,goodnormal),('normal_vs_U600_deg',delta,common)]:
                row.update({f'{label}_{k}':v for k,v in stats(values[view][support[view]]).items()})
            for label,values in [('distal_arm_depth_abs',d),('distal_arm_Direct',direct)]:
                row.update({f'{label}_{k}':v for k,v in stats(values[view][target_arm[view]]).items()})
            for name,(yl,yh) in bands.items():
                fixed=valid[view]&(target_world[view,...,1]>=yl)&(target_world[view,...,1]<=yh)
                row.update({f'{name}_depth_abs_{k}':v for k,v in stats(d[view][fixed]).items()})
                fixedfg=fg[view]&(ref['world_points'][0,view,...,1]>=yl)&(ref['world_points'][0,view,...,1]<=yh)
                row[f'{name}_fixed_U600_FG_n']=int(fixedfg.sum())
                row[f'{name}_normalvalid_n']=int((fixedfg&nv[view]).sum())
                sf=fixedfg[::4,::4]&stride[view]
                row[f'{name}_stride4_n']=int(sf.sum())
            allrows.append(row)
        base=raw['indicator_base'][0];final=raw['indicator'][0];res=np.abs(final-base)
        model['field_residual_abs']=stats(res.ravel());model['field_sign_flip_n']=int(((base<0)!=(final<0)).sum())
        model['field_base_range']=[float(base.min()),float(base.max())]
        model['field_final_range']=[float(final.min()),float(final.max())]
        center=a['unit_center'][0];scale=float(a['unit_scale'][0]);axes=[(np.arange(128)/128-.5)*scale+center[i] for i in range(3)]
        voxelbox=((axes[0][:,None,None]>=arm_box[0][0])&(axes[0][:,None,None]<=arm_box[1][0])&
                  (axes[1][None,:,None]>=arm_box[0][1])&(axes[1][None,:,None]<=arm_box[1][1])&
                  (axes[2][None,None,:]>=arm_box[0][2])&(axes[2][None,None,:]<=arm_box[1][2]))
        model['distal_arm']['field_residual_abs']=stats(res[voxelbox])
        model['distal_arm']['field_sign_flip_n']=int((((base<0)!=(final<0))&voxelbox).sum())
        cells=[(np.arange(127)+.5)/128*scale+center[i]-.5*scale for i in range(3)]
        cellbox=((cells[0][:,None,None]>=arm_box[0][0])&(cells[0][:,None,None]<=arm_box[1][0])&
                 (cells[1][None,:,None]>=arm_box[0][1])&(cells[1][None,:,None]<=arm_box[1][1])&
                 (cells[2][None,None,:]>=arm_box[0][2])&(cells[2][None,None,:]<=arm_box[1][2]))
        model['distal_arm']['base_zero_cross_cells']=int((crossing(base)&cellbox).sum())
        model['distal_arm']['final_zero_cross_cells']=int((crossing(final)&cellbox).sum())
        gy=(np.arange(base.shape[1])/base.shape[1]-.5)*float(a['unit_scale'][0])+float(a['unit_center'][0,1])
        for name,(yl,yh) in bands.items():
            support=valid&(target_world[...,1]>=yl)&(target_world[...,1]<=yh)
            supportfg=fg&(ref['world_points'][0,...,1]>=yl)&(ref['world_points'][0,...,1]<=yh)
            samp=supportfg[:,::4,::4]&stride
            fieldsel=(gy>=yl)&(gy<=yh)
            model['local'][name]={'depth':stats(d[support]),'direct':stats(direct[support]),
                                 'fixed_U600_FG_n':int(supportfg.sum()),'normalvalid_n':int((supportfg&nv).sum()),
                                 'stride4_n':int(samp.sum()),'field_residual_abs':stats(res[:,fieldsel,:].ravel()),
                                 'field_sign_flip_n':int(((base[:,fieldsel,:]<0)!=(final[:,fieldsel,:]<0)).sum())}
        meshes[phase]={}
        for stage,filename in [('BASE','base_mesh.ply'),('FINAL','clothed_mesh.ply')]:
            m=trimesh.load(Path(records[phase]['original_complete_RAW']).parent/filename,process=False,maintain_order=True)
            meshes[phase][stage]=(np.asarray(m.vertices),np.asarray(m.faces))
            model[stage]={'vertices':len(m.vertices),'faces':len(m.faces)}
            model['distal_arm'][stage+'_vertices_n']=int(inside(np.asarray(m.vertices),arm_box).sum())
        summary['models'][phase]=model
        for row in allrows[-6:]:
            row['model_global_field_sign_flip_n']=model['field_sign_flip_n']
            for key,value in model['field_residual_abs'].items():row['model_global_field_residual_abs_'+key]=value
            for key,value in model['distal_arm'].items():
                if isinstance(value,dict):
                    for k,v in value.items():row['model_distal_arm_'+key+'_'+k]=v
                else:row['model_distal_arm_'+key]=value
        del raw
    OUT.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in allrows for k in row))
    with (OUT/'0019_关键指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
    # 全形以四模型全RAW和所有面联合范围定窗；局部prior固定，不逐模型跟踪。
    allp=np.concatenate(list(rawpoints.values())+[v for group in meshes.values() for v,f in group.values()])
    boxes={}
    for az in [0,90]:
        proj=project(allp,az);lo=proj[:,:2].min(0);hi=proj[:,:2].max(0)
        extent=float(max(hi-lo)*1.05);mid=(lo+hi)/2
        boxes[('FULL',az)]=(mid[0]-extent/2,mid[0]+extent/2,mid[1]-extent/2,mid[1]+extent/2)
        pr=project(prior,az);left,right=pr[:,0].min(),pr[:,0].max();mid=(left+right)/2;span=(right-left)*1.2
        for name,(yl,yh) in bands.items():boxes[(name,az)]=(mid-span/2,mid+span/2,-yh,-yl)
    summary['display_boxes']={f'{name}_{az}':list(box) for (name,az),box in boxes.items()}
    (AUX/'0019_cpu_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    if (OUT/'0019_局部对照.png').exists():
        print(json.dumps(summary['models'],ensure_ascii=False,indent=2,default=float))
        return
    cellw,cellh=320,280;header=110;rowheight=640
    sheet=Image.new('RGB',(12*cellw,header+4*rowheight+115),'white');draw=ImageDraw.Draw(sheet)
    draw.text((12,8),'0019 / 0019_06 / 0224 | SAVED ARTIFACTS ONLY | CPU | shared orthographic windows',font=FONT,fill='black')
    draw.text((12,35),'Each stage: FULL / HEAD-NECK band / ARM band. Each tile: 0 deg above, 90 deg below. Negative Y is up.',font=FONT,fill='black')
    stages=['RGB RAW (all FG)','SAVED STRIDE4 (all)','BASE (all faces)','FINAL (all faces)']
    for s,stage in enumerate(stages):
        draw.text((s*3*cellw+12,66),stage,font=FONT,fill='black')
        for j,name in enumerate(['FULL']+list(bands)):
            draw.text(((s*3+j)*cellw+12,89),name,font=FONT,fill='black')
    for i,phase in enumerate(PHASES):
        y=header+i*rowheight
        m=summary['models'][phase]
        draw.rectangle((0,y,12*cellw,y+31),fill=(224,231,239))
        draw.text((12,y+5),f'{phase} | all foreground {m["foreground_n"]}; saved stride4 {m["stride4_n"]}; BASE {m["BASE"]["faces"]} faces; FINAL {m["FINAL"]["faces"]} faces',font=FONT,fill='black')
        for s in range(4):
            if s==0:pts=rawpoints[phase];cols=colors[phase];faces=None
            elif s==1:pts=samplepoints[phase];cols=np.tile(np.array([[50,100,145]],np.uint8),(len(pts),1));faces=None
            else:pts,faces=meshes[phase][['BASE','FINAL'][s-2]];cols=None
            for j,name in enumerate(['FULL']+list(bands)):
                for n,az in enumerate([0,90]):
                    pic=panel(pts,cols,faces,az,boxes[(name,az)],(cellw,cellh))
                    if az==90:
                        l,r,b,t=boxes[(name,az)];factor=min((cellw-12)/(r-l),(cellh-24)/(t-b))
                        box=(arm_box[0][2],arm_box[1][2],-arm_box[1][1],-arm_box[0][1])
                        x1=(box[0]-(l+r)/2)*factor+cellw/2;x2=(box[1]-(l+r)/2)*factor+cellw/2
                        y1=((b+t)/2-box[3])*factor+(cellh+16)/2;y2=((b+t)/2-box[2])*factor+(cellh+16)/2
                        ImageDraw.Draw(pic).rectangle((x1,y1,x2,y2),outline=(210,45,45),width=1)
                    sheet.paste(pic,((s*3+j)*cellw,y+34+n*cellh))
    y=header+4*rowheight+10
    draw.text((12,y),'Local crops are prior-height bands, not anatomy GT. Red box = frozen distal-arm space. No confidence/component deletion; depth order only for display.',font=FONT,fill='black')
    draw.text((12,y+26),'GT/teacher normal is unavailable. BT/BTF have no direct teacher supervision for 0019; differences reflect shared parameter updates.',font=FONT,fill='black')
    draw.text((12,y+52),'All original NPZ/PLY remain local, read-only. No network/model/DPSR/mesh extraction was run. Metrics use original camera-Z/world units.',font=FONT,fill='black')
    sheet.save(OUT/'0019_局部对照.png',optimize=True)
    print(json.dumps(summary['models'],ensure_ascii=False,indent=2,default=float))


if __name__=='__main__':
    main()
