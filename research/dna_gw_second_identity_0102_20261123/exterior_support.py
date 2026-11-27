"""检查原鞋袍世界盒内，学生六源首面能否绑定独立物理量测。"""
import csv
from pathlib import Path
import h5py
import numpy as np
import open3d as o3d

from independent_rays import intersect,make_rays
from qa_protocol import (AUX,MODELS,ORIGINAL,OUT,QA,SOURCE,cameras,grid_band,original_mesh,project,
                         read,save,table)


def rows(path):
    """读回含前导零与未测列的证据记录。"""
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def load_mesh(path):
    """以原 PLY 三角顺序建立 CPU 索引，保留世界面法线。"""
    vertices,faces=original_mesh(path)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
    return scene,winding_normals(vertices,faces)


def winding_normals(vertices,faces):
    """按原三角绕序计算世界单位法线，零面积面保留零向量。"""
    point=vertices[faces]
    normal=np.cross(point[:,1]-point[:,0],point[:,2]-point[:,0])
    length=np.linalg.norm(normal,axis=1)
    np.divide(normal,length[:,None],out=normal,where=length[:,None]>0)
    return normal


def aux_review(models,candidate,material):
    """在原 AUX 像素检查混合三视旧点处各模型的前层与相对深度。"""
    records,detail=[],[]
    cb=cameras()
    xyz=np.array([[float(v[x]) for x in ("x_m","y_m","z_m")] for v in candidate])
    for cam in AUX:
        c=cb[cam]
        with Path(c["annotation_smc"]).open("rb") as src,h5py.File(src,"r") as f:
            calibration=f[f"Camera_Parameter/{cam:02}"]
            k,d,rt=[calibration[x][...].astype(float) for x in ("K","D","RT")]
        xy,projected_z=project(xyz,k,d,rt)
        rays,_=make_rays(xy,k,d,rt)
        for name,scene in models.items():
            hit,face=intersect(scene,rays)
            dz=(hit-projected_z)*1000
            finite=np.isfinite(dz)
            records.append(dict(camera=cam,model=name,observation=material[cam],legacy_positions=len(candidate),
                               finite_model_first_hit=int(finite.sum()),no_hit=int((~finite).sum()),
                               candidate_minus_model_Z_median_mm=float(-np.median(dz[finite])) if finite.any() else None,
                               abs_difference_P95_mm=float(np.percentile(np.abs(dz[finite]),95)) if finite.any() else None,
                               count_within_one_0102_grid=int((np.abs(dz)<=grid_band()*1000).sum()),
                               independent_AUX_material_positions=0,independent_AUX_depth=0,independent_AUX_normal=0,
                               coordinate_producer="35,37 BUILD; 36 AUX C-test",new_GW_training_role="AUX6_RESERVED",
                               NeuS2_historical_role="TRAIN40",support="MATERIAL_LAYER_UNKNOWN"))
            for idx in range(len(xyz)):
                detail.append(dict(camera=cam,model=name,candidate_id=candidate[idx]["candidate_id"],
                        native_u=xy[idx,0],native_v=xy[idx,1],candidate_Z_m=projected_z[idx],
                        model_first_Z_m=hit[idx] if finite[idx] else None,face_id=int(face[idx]),
                        model_minus_candidate_mm=dz[idx] if finite[idx] else None,
                        observation=material[cam],same_layer_certified=False))
    table(OUT/"GW_0102_AUX6_GEOMETRY_QA.csv",records)
    table(QA/"AUX6_26_CANDIDATE_MODEL_FRONT_LAYER.csv",detail)


def native_maps():
    """将独立传感器读数、世界坐标和新网格面号放回原设备像素。"""
    maps=[]
    for i in range(8):
        with np.load(QA/f"KINECT{i}_FIRST_HITS.npz") as data:
            xy=data["pixel_xy"]
            shape=tuple(data["shape_hw"])
            xyz=np.full((*shape,3),np.nan,"f4")
            raw=np.full(shape,np.nan,"f4")
            first=np.full(shape,np.nan,"f4")
            stable=np.zeros(shape,bool)
            face=np.full(shape,-1,"i4")
            pos=xy[:,1],xy[:,0]
            xyz[pos]=data["world_measured"]
            raw[pos]=data["raw_depth_mm"].astype("f4")*.001
            first[pos]=data["GW_OFFICIAL_FULL_z_m"]
            face[pos]=data["GW_OFFICIAL_FULL_face"]
            stable[pos]=data["valid_3x3"]
            maps.append((data["K"].copy(),data["D"].copy(),data["RT"].copy(),raw,stable,first,face,xyz))
    return maps


def witnesses(world,source_face,devices):
    """按设备核实三维接近、原射线首面与同面号，保留可靠前层矛盾。"""
    n=len(world)
    near=np.zeros(n,"u1")
    same=np.zeros(n,"u1")
    conflict=np.zeros(n,"u1")
    foreground_before=np.zeros(n,"u1")
    projected=np.full((n,8,2),-1,"i2")
    device_dist=np.full((n,8),np.nan,"f4")
    if n==0:
        return near,same,conflict,foreground_before,projected,device_dist
    for i,(k,d,rt,raw,stable,depth,face,measured) in enumerate(devices):
        uv,z=project(world,k,d,rt)
        xy=np.rint(uv).astype(int)
        h,w=raw.shape
        inside=(z>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ids=np.flatnonzero(inside)
        x,y=xy[ids].T
        projected[ids,i]=xy[ids]
        sensor_z=raw[y,x]
        same_ray=(face[y,x]>=0)&(np.abs(depth[y,x]-z[ids])<=grid_band())
        delta=z[ids]-sensor_z
        dist=np.linalg.norm(world[ids]-measured[y,x],axis=1)
        device_dist[ids,i]=dist
        good=stable[y,x]&same_ray&(np.abs(delta)<=grid_band())&(dist<=grid_band())
        near[ids[good]]|=1<<i
        match=good&(face[y,x]==source_face[ids])
        same[ids[match]]|=1<<i
        inconsistent=stable[y,x]&same_ray&(delta < -grid_band())
        conflict[ids[inconsistent]]|=1<<i
        obscures=stable[y,x]&same_ray&(delta > grid_band())
        foreground_before[ids[obscures]]|=1<<i
    return near,same,conflict,foreground_before,projected,device_dist


def colour_footprint(scene,cam,xy,source_faces,source_z):
    """复核 AREA 颜色中心及其四角的物理射线和实际原 Mask。"""
    c=cameras()[cam]
    binding=next(r for r in rows(OUT/"0102_STUDENT_CAMERA_BINDING.csv") if int(r["camera"])==cam)
    # 原学生监督采用整数 K；此处单独按原 RGB 的面积采样反算物理位置。
    result={}
    with Path(c["annotation_smc"]).open("rb") as src,h5py.File(src,"r") as f:
        import cv2
        mask=cv2.imdecode(f[c["mask_key"]][...],cv2.IMREAD_UNCHANGED)
        mask=mask if mask.ndim==2 else mask.max(2)
        group=f[f"Camera_Parameter/{cam:02}"]
        k,d,rt=[group[x][...].astype(float) for x in ("K","D","RT")]
    if not len(xy):
        return dict(photo_probe_face=np.empty((0,5),"i4"),photo_probe_Z_m=np.empty((0,5),"f4"),
                    photo_probe_mask=np.empty((0,5),bool),photo_center_same_face=np.empty(0,bool),
                    photo_footprint_continuous=np.empty(0,bool))
    h,w=mask.shape
    scale=np.array([433/w,518/h])
    base=xy.astype(float)-np.array([int(binding["left"]),0])
    fs,zs,ms=[],[],[]
    for offset in [(0.5,0.5),(0,0),(1,0),(0,1),(1,1)]:
        ideal=(base+offset)/scale-.5
        rays,dirs=make_rays(ideal,k,np.zeros(5),rt)
        zz,ff=intersect(scene,rays)
        fs.append(ff);zs.append(zz)
        distortion,unused_z=project(rt[:3,3]+dirs,k,d,rt)
        uv=np.rint(distortion).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        support=np.zeros(len(uv),bool)
        support[inside]=mask[uv[inside,1],uv[inside,0]]>0
        ms.append(support)
    faces=np.stack(fs,axis=1);depths=np.stack(zs,axis=1);foreground=np.stack(ms,axis=1)
    result.update(photo_probe_face=faces,photo_probe_Z_m=depths,photo_probe_mask=foreground,
                  photo_center_same_face=faces[:,0]==source_faces,
                  photo_footprint_continuous=np.all(np.abs(depths-source_z[:,None])<=grid_band(),axis=1)&foreground.all(1))
    return result


def source_targets(scene,normals):
    """在六个原网格上保留真实测量、网格预测、逐设备见证和层未知。"""
    dest=OUT/"GW_0102_TEACHER_TARGETS"
    dest.mkdir(exist_ok=True)
    bounds=np.asarray(read(QA/"FROZEN_QA_PROTOCOL.json")["region_boxes"]["RIGHT_SHOE_ROBE"])
    dev=native_maps()
    original_winners=rows(OUT/"0102_KINECT_ROI_ORIGINAL_SIX_SOURCE_WINNERS.csv")
    with np.load(OUT/"0102_KINECT_ORIGINAL_SIX_SOURCE_WINNER_KEYS.npz") as archive:
        prior_bits=archive["original_M_source_bits"]
        prior_mm=archive["raw_depth_mm"]
    stats=[]
    review=[]
    with np.load(ORIGINAL/"INPUTS.npz") as inputs,np.load(ORIGINAL/"TARGETS.npz") as labels:
        all_k=inputs["K"][0]
        all_e=inputs["E"][0]
        masks=inputs["masks"][0].astype(bool)
        depth=labels["depth"][0]
        valid=labels["depth_valid"][0].astype(bool)
    for slot,cam in enumerate(SOURCE):
        k=all_k[slot].astype(float);e=all_e[slot].astype(float);rt=np.linalg.inv(e)
        fg=masks[slot]
        yy,xx=np.nonzero(fg)
        rays,dirs=make_rays(np.column_stack((xx,yy)),k,np.zeros(5),rt)
        z,face=intersect(scene,rays)
        world=rt[:3,3]+dirs*z[:,None]
        region=np.all((world>=bounds[0])&(world<=bounds[1]),axis=1)
        w=world[region]
        src_face=face[region]
        near,same,conflict,foreground_before,projected,dist=witnesses(w,src_face,dev)
        xy=np.column_stack((xx[region],yy[region]))
        pos=xy[:,1],xy[:,0]
        independent_m=fg&valid[slot]
        m=independent_m[pos]
        footprint=colour_footprint(scene,cam,xy,src_face,z[region])
        selected=[r for r in original_winners if int(r["student_camera"])==cam]
        measured_roi=np.zeros(fg.shape,bool)
        for r in selected:
            measured_roi[int(r["xf_v"]),int(r["xf_u"])]=True
        prior_depth=depth[slot][pos]
        local_src_conflict=m&(np.abs(prior_depth-z[region])>grid_band())
        diagnostic_agreement=(same!=0)&(conflict==0)&~local_src_conflict
        for idx in np.flatnonzero(diagnostic_agreement):
            for sensor in range(8):
                if same[idx] & (1<<sensor):
                    pu,pv=map(int,projected[idx,sensor])
                    raw_mm=int(round(dev[sensor][3][pv,pu]*1000))
                    old_bits=int(prior_bits[sensor,pv,pu])
                    if old_bits and int(prior_mm[sensor,pv,pu])!=raw_mm:
                        raise ValueError("同一原设备像素的历史有效毫米读数不一致")
                    review.append(dict(cam=cam,source_u=int(xy[idx,0]),source_v=int(xy[idx,1]),
                        original_M=bool(m[idx]),original_U=not bool(m[idx]),world_x=w[idx,0],world_y=w[idx,1],world_z=w[idx,2],
                        model_face=int(src_face[idx]),Kinect=sensor,raw_u=pu,raw_v=pv,
                        raw_Z_mm=raw_mm,original_M_any_source_bits=old_bits,distance_mm=dist[idx,sensor]*1000,
                        photo_center_same_face=bool(footprint["photo_center_same_face"][idx]),
                        photo_footprint_continuous=bool(footprint["photo_footprint_continuous"][idx]),
                        same_face_device_bits=int(same[idx]),reliable_mesh_in_front_conflict=False))
        np.savez_compressed(dest/f"cam{cam:02}_ROI_SOURCE_EVIDENCE.npz",
                physical_camera=cam,student_slot=slot,K=k,E_w2c=e,shape_hw=fg.shape,
                original_M=independent_m,original_M_camera_Z_m=np.where(independent_m,depth[slot],0),
                raw_measured_world_in_roi=measured_roi,foreground=fg,
                source_grid_ROI_xy=xy.astype("u2"),GW_face_id=src_face,GW_first_Z_m=z[region],GW_world_m=w.astype("f4"),
                MESH_DERIVED_NORMAL_world=normals[src_face],observed_on_original_M=m,
                native_proximity_device_bits=near,native_same_face_device_bits=same,
                native_reliable_behind_GW_conflict_bits=conflict,native_foreground_before_GW_bits=foreground_before,sensor_raw_uv=projected,
                distance_to_native_measured_m=dist,original_grid_depth_layer_conflict=local_src_conflict,
                numeric_same_face_no_conflict=diagnostic_agreement,**footprint)
        stats.append(dict(cam=cam,original_source_F=int(fg.sum()),original_source_M=int(independent_m.sum()),
                original_source_U=int((fg&~independent_m).sum()),original_M_world_in_ROI=len(selected),
                original_M_stable_in_ROI=sum(r["raw_stable_3x3"]=="True" for r in selected),new_GW_hit_pos_in_ROI=len(w),
                GW_M=int(m.sum()),GW_U=int((~m).sum()),native_proximity=int((near!=0).sum()),
                native_same_face=int((same!=0).sum()),native_reliable_behind_conflict=int((conflict!=0).sum()),
                other_raw_foreground_in_front_of_predicted_surface=int((foreground_before!=0).sum()),
                original_M_depth_conflict=int(local_src_conflict.sum()),
                stable_same_face_no_conflict_M=int((m&diagnostic_agreement).sum()),
                stable_same_face_no_conflict_U=int((~m&diagnostic_agreement).sum()),
                material_probe_U=int((~m&diagnostic_agreement&footprint["photo_center_same_face"]&footprint["photo_footprint_continuous"]).sum()),
                stable_two_device_same_face_no_conflict_U=int((~m&diagnostic_agreement&((same&(same-1))!=0)).sum()),
                unmatched_GW_only_U=int((~m&(near==0)&(conflict==0)).sum())))
    table(QA/"SIX_SOURCE_ROI_WITNESS_COUNTS.csv",stats)
    csv_fields=["cam","source_u","source_v","original_M","original_U","world_x","world_y","world_z",
                "model_face","Kinect","raw_u","raw_v","raw_Z_mm","original_M_any_source_bits","distance_mm","photo_center_same_face",
                "photo_footprint_continuous","same_face_device_bits","reliable_mesh_in_front_conflict"]
    with (QA/"SIX_SOURCE_ROI_PHYSICAL_WITNESS_REVIEW.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(review)
    return stats


def main(scenes=None):
    """汇总辅助观察、原传感器与六源证据，供局部目标资格裁定。"""
    candidates=rows(OUT/"0102_EXISTING_CANDIDATES.csv")
    material={3:"white robe nearer surface",5:"black hem nearer surface",18:"robe/green boundary; toe unrecognized",
              27:"toe upper/sole edge",32:"frontal toe upper/pale sole edge",36:"oblique toe upper/sole edge"}
    if scenes is None:
        scenes={}
        for name,path in MODELS.items():
            scenes[name],face_normals=load_mesh(path)
            if name=="GW_OFFICIAL_FULL":
                normals=face_normals
    else:
        normals=winding_normals(*original_mesh(MODELS["GW_OFFICIAL_FULL"]))
    aux_review(scenes,candidates,material)
    summary=source_targets(scenes["GW_OFFICIAL_FULL"],normals)
    save(QA/"SOURCE_ROI_EVIDENCE_RECEIPT.json",dict(rows=summary,geometry="original meters",
         six_source="undistorted student 518x518 integer centres, original K/E",
         native_relations="same face requires source and sensor first triangle IDs equal and XYZ distance <= one actual grid",
         physical_label_source="individual physical material review is separate from numerical agreement",surface_normal="original face winding only"))


if __name__=="__main__":
    main()
