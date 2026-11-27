"""同世界展示四个整场几何、右鞋袍局部及原生设备首层证据。"""
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont

from research.dna_gw40_dngw0019_replay_20261125.paths import MONO_FONT
import cv2

from qa_protocol import FIG,MODELS,OLD,ORIGINAL,OUT,QA,RUN,SOURCE,grid_band,project_box,read


def font(size):
    """使用本机投影可读的字体。"""
    return ImageFont.truetype(str(MONO_FONT),size)


def ortho(scene,bounds,ang,width,height,near):
    """从共同的正交世界相机读取真实首面，以原面方向作几何着色。"""
    low,high=np.asarray(bounds)
    center=(low+high)*.5
    view=np.array([np.sin(ang),0,np.cos(ang)])
    right=np.array([np.cos(ang),0,-np.sin(ang)])
    down=np.array([0,1,0])
    scale=max((high[1]-low[1])/height,
              np.dot(np.abs(right),high-low)/width)*1.08
    xx,yy=np.meshgrid((np.arange(width)-(width-1)*.5)*scale,(np.arange(height)-(height-1)*.5)*scale)
    rays=np.empty((height,width,6),"f4")
    rays[:,:,:3]=center-near*view+xx[:,:,None]*right+yy[:,:,None]*down
    rays[:,:,3:]=view
    out=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
    finite=np.isfinite(out["t_hit"].numpy())
    normal=out["primitive_normals"].numpy()
    light=-.75*view-.36*right-.52*down
    shade=.3+.7*np.abs(normal@light)
    image=np.full((height,width,3),245,"u1")
    image[finite]=np.clip(shade[finite,None]*np.array([106,165,194])+15,0,255).astype("u1")
    return Image.fromarray(image)


def mesh_panels(scenes):
    """完整场景、人体观察尺度与鞋袍分别固定窗口，三角面不裁剪。"""
    sources=read(QA/"MESH_WORLD_SOURCES.json")
    sizes=np.array([[v["world_min"],v["world_max"]] for v in sources.values()])
    extent=[sizes[:,0].min(0),sizes[:,1].max(0)]
    with np.load(OLD/"source_fitted/SOURCE_FITTED.npz") as f:
        v=f["vertices_world"]
        body=[v.min(0),v.max(0)]
    shoe=read(QA/"FROZEN_QA_PROTOCOL.json")["region_boxes"]["RIGHT_SHOE_ROBE"]
    far=2*np.max(np.abs(extent))+np.max(np.abs(body))+1
    for name,box,h in [("WHOLE_SCENE",extent,365),("BODY",body,365),("RIGHT_SHOE_ROBE",shoe,205)]:
        w=305
        canvas=Image.new("RGB",(4*w,3*(h+27)+58),"white")
        pen=ImageDraw.Draw(canvas)
        pen.text((9,5),f"0102_02 / 0000    {name}    world metres | identical orthographic views | all mesh components",fill="black",font=font(19))
        for col,(model,scene) in enumerate(scenes.items()):
            pen.text((col*w+9,34),model,fill="black",font=font(18))
            for i,deg in enumerate((0,90,180)):
                top=58+i*(h+27)
                canvas.paste(ortho(scene,box,np.deg2rad(deg),w,h,far),(col*w,top))
                pen.text((col*w+9,top+h+2),f"{deg} deg",font=font(14),fill="black")
        canvas.save(FIG/f"FOUR_MESHES_{name}.png")


def sensor_panel():
    """按原设备像素显示稳定射线上的相容、前后冲突与漏面。"""
    colors=np.array([[250,250,250],[170,170,170],[41,160,83],[212,65,54],[60,112,205],[230,165,55]],"u1")
    canvas=Image.new("RGB",(1170,730),"white")
    pen=ImageDraw.Draw(canvas)
    pen.text((8,5),"0102_02 / 0000   original Kinect pixels | GW official full first hit",font=font(23),fill="black")
    pen.text((8,37),"Green: within band   red: near   blue: far   ochre: no hit   gray: unstable original depth",font=font(17),fill="black")
    band=grid_band()*1000
    for i in range(8):
        with np.load(QA/f"KINECT{i}_FIRST_HITS.npz") as z:
            x,y=z["pixel_xy"].T
            mask=z["valid_3x3"]
            diff=z["GW_OFFICIAL_FULL_z_m"]*1000-z["raw_depth_mm"].astype(float)
            status=np.ones(len(x),"u1")
            status[mask]=5
            status[mask&(np.abs(diff)<=band)]=2
            status[mask&(diff< -band)]=3
            status[mask&(diff>band)]=4
            arr=np.zeros(tuple(z["shape_hw"]),"u1")
            arr[y,x]=status
        pic=Image.fromarray(colors[arr]).crop((max(0,int(x.min())-5),max(0,int(y.min())-5),int(x.max())+6,int(y.max())+6))
        pic.thumbnail((278,292))
        left=i%4*292;top=65+i//4*331
        canvas.paste(pic,(left+(292-pic.width)//2,top+22))
        pen.text((left+8,top),f"K{i}    {len(x):,} measurements",font=font(14),fill="black")
    canvas.save(FIG/"GW_KINECT8_MEASURED_RAYS.png")


def source_panel():
    """在六个真实学生 RGB 局部并排标出数值见证、冲突及预测。"""
    bounds=read(QA/"FROZEN_QA_PROTOCOL.json")["region_boxes"]["RIGHT_SHOE_ROBE"]
    with np.load(ORIGINAL/"INPUTS.npz") as src:
        imgs=src["images"][0].transpose(0,2,3,1)
        cams=src["K"][0]
        es=src["E"][0]
    canvas=Image.new("RGB",(3*380,2*207+75),"white")
    pen=ImageDraw.Draw(canvas)
    pen.text((7,4),"0102   original six student photos | right shoe / robe ROI | original and GW first surface",font=font(20),fill="black")
    pen.text((7,33),"Green: same face and native depth within band   red: depth conflict   blue: model-only or unverified",font=font(16),fill="black")
    for i,cam in enumerate(SOURCE):
        img=np.uint8(np.clip(np.rint(255*imgs[i]),0,255))
        h,w=img.shape[:2]
        box=project_box(bounds,cams[i],np.zeros(5),np.linalg.inv(es[i]),h,w)
        mark=img.copy()
        with np.load(OUT/f"GW_0102_TEACHER_TARGETS/cam{cam:02}_ROI_SOURCE_EVIDENCE.npz") as z:
            xy=z["source_grid_ROI_xy"]
            good=z["numeric_same_face_no_conflict"]
            bad=(z["native_reliable_behind_GW_conflict_bits"]!=0)|z["original_grid_depth_layer_conflict"]
            for mask,color in [(~good&~bad,[60,100,245]),(bad,[245,50,45]),(good,[20,224,99])]:
                p=xy[mask]
                mark[p[:,1],p[:,0]]=(.42*img[p[:,1],p[:,0]]+.58*np.array(color)).astype("u1")
        left=i%3*380;top=65+i//3*207
        pen.text((left+5,top),f"cam {cam:02}   original | numeric layer evidence",font=font(14),fill="black")
        for col,a in enumerate((img,mark)):
            patch=Image.fromarray(a).crop(box)
            ratio=min(180/patch.width,165/patch.height)
            patch=patch.resize((max(1,round(patch.width*ratio)),max(1,round(patch.height*ratio))),Image.Resampling.NEAREST)
            canvas.paste(patch,(left+col*187+(180-patch.width)//2,top+29))
    canvas.save(FIG/"SIX_SOURCE_ORIGINAL_RGB_AND_GW_LAYER_EVIDENCE.png")


def field_panel():
    """用 BUILD35/37 官方已保存的字段绘制鞋袍区域，不调用渲染器。"""
    bounds=read(QA/"FROZEN_QA_PROTOCOL.json")["region_boxes"]["RIGHT_SHOE_ROBE"]
    canvas=Image.new("RGB",(1130,520),"white")
    pen=ImageDraw.Draw(canvas)
    pen.text((8,6),"0102 | official frozen Gaussian field | training cameras 35, 37 | right shoe / robe window",font=font(20),fill="black")
    for i,cam in enumerate([35,37]):
        with np.load(RUN/f"GAUSSIAN_FIELD_BUILD34/cam_{cam:02}.npz") as data:
            rgb=np.moveaxis(data["rgb"],0,-1)
            alpha=data["alpha"]
            depth=data["median_camera_z_m"]
            normal=np.moveaxis(data["normal"],0,-1)
            h,w=depth.shape
            box=project_box(bounds,data["K"],np.zeros(5),np.linalg.inv(data["E"]),h,w)
        x0,y0,x1,y1=box
        patch=depth[y0:y1,x0:x1]
        z8=np.uint8(np.clip((patch-2)/2,0,1)*255)
        zimg=cv2.cvtColor(cv2.applyColorMap(z8,cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
        zimg[patch<=0]=0
        arrays=[(rgb[y0:y1,x0:x1],"official RGB"),(alpha[y0:y1,x0:x1],"alpha [0,1]"),
             (zimg,"median Z 2-4 m"),(.5+.5*normal[y0:y1,x0:x1],"mesh-free rendered normal")]
        for j,(a,label) in enumerate(arrays):
            if a.dtype!=np.uint8:
                a=(np.clip(a,0,1)*255).astype("u1")
            patch_image=Image.fromarray(a)
            ratio=min(270/patch_image.width,168/patch_image.height)
            patch_image=patch_image.resize((max(1,round(patch_image.width*ratio)),max(1,round(patch_image.height*ratio))),Image.Resampling.NEAREST)
            x=4+j*280;y=48+i*230
            pen.text((x,y),f"BUILD{cam} {label}",font=font(14),fill="black")
            canvas.paste(patch_image,(x+(270-patch_image.width)//2,y+25))
    canvas.save(FIG/"OFFICIAL_GAUSSIAN_FIELD_BUILD35_37.png")
