"""完整彩色RAW在共同物理观察坐标中显示，图像遮挡不改变点云。"""
import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.asarray([[230,70,60],[50,150,235],[40,180,95],
                      [225,170,35],[170,80,215],[35,185,185]], np.uint8)


def project(points, angle):
    """原DNA世界以负Y向上，仅旋转显示相机而不修改坐标。"""
    radians = np.deg2rad(angle)
    x = points[:,0]*np.cos(radians)+points[:,2]*np.sin(radians)
    z = -points[:,0]*np.sin(radians)+points[:,2]*np.cos(radians)
    return np.column_stack([x,-points[:,1],z])


def panel(points, colors, angle, bounds, label, size=(600,650)):
    """共同正交相机与一点一像素绘图，按深度处理显示遮挡。"""
    width,height = size
    p = project(points,angle)
    left,right,bottom,top = bounds
    factor = min((width-20)/(right-left),(height-50)/(top-bottom))
    x = np.rint((p[:,0]-(left+right)/2)*factor+width/2).astype(int)
    y = np.rint(((bottom+top)/2-p[:,1])*factor+(height+30)/2).astype(int)
    valid = (x>=0)&(x<width)&(y>=30)&(y<height)&np.isfinite(p).all(1)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(p[indices,2],kind='stable')]
    canvas = np.full((height,width,3),245,np.uint8)
    canvas[y[indices],x[indices]]=colors[indices]
    result = Image.fromarray(canvas)
    ImageDraw.Draw(result).text((8,7),label,fill='black')
    return result


def make_figures(records, out, prior):
    """三模型共用全量联合范围和先验固定局部范围，不按结果重新取景。"""
    angles=[0,90,180]
    all_points=np.concatenate([r['points'] for r in records])
    projected=[project(all_points,a) for a in angles]
    horizontal=max(float(np.ptp(p[:,0])) for p in projected)
    centers=[float((p[:,0].min()+p[:,0].max())/2) for p in projected]
    bottom=min(float(p[:,1].min()) for p in projected)
    top=max(float(p[:,1].max()) for p in projected)
    extent=max(horizontal,top-bottom)*1.06
    middle=(bottom+top)/2
    bounds=[(c-extent/2,c+extent/2,middle-extent/2,middle+extent/2) for c in centers]
    index=[]
    for mode in ['RGB','SOURCE_COLOR']:
        sheet=Image.new('RGB',(1800,650*len(records)),'white')
        for row,record in enumerate(records):
            colors=record['colors'] if mode=='RGB' else PALETTE[record['view_id']]
            for col,angle in enumerate(angles):
                view=panel(record['points'],colors,angle,bounds[col],
                           f"{record['name']} | {angle} deg | {mode} | full FG {len(colors)}")
                sheet.paste(view,(col*600,row*650))
        path=out/f'RAW_{mode}_0_90_180.png';sheet.save(path);index.append(str(path))
    # 局部只限定展示窗口；NPZ/PLY始终包含完整六视图前景。
    low,high=prior.min(0),prior.max(0)
    height=float(high[1]-low[1]);top=-float(low[1])
    bands=[('HEAD',-.03,.23),('CHEST',.20,.52),('ARMS',.18,.66),
           ('LEGS',.50,.92),('SHOES',.86,1.08)]
    for label,start,end in bands:
        sheet=Image.new('RGB',(1800,420*len(records)),'white')
        for row,record in enumerate(records):
            for col,angle in enumerate(angles):
                p=project(prior,angle);l,r=float(p[:,0].min()),float(p[:,0].max())
                span=(r-l)*1.20;mid=(l+r)/2
                box=(mid-span/2,mid+span/2,top-end*height,top-start*height)
                view=panel(record['points'],record['colors'],angle,box,
                           f"{record['name']} | {label} fixed prior band | {angle} deg",(600,420))
                sheet.paste(view,(col*600,row*420))
        path=out/f'RAW_DETAIL_{label}.png';sheet.save(path);index.append(str(path))
    return dict(paths=index,angles=angles,world_up=[0,-1,0],point_size_px=1,
                full_bounds=bounds,local_bands=bands,
                local_scope='Fixed prior-height display bands; not anatomical segmentation. Full arrays retain every foreground point.',
                visibility='Depth order only for display; no confidence, measurement-validity or component filtering.')
