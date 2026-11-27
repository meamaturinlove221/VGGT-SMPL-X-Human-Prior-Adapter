# VGGT × SMPL-X 六视图与 GW40 / DN_GW 重放

流水线从同帧标定影像和 SMPL-X 先验出发。GaussianWrapping 在 40 路 DNA-Rendering RGB 上重建同帧教师网格，BODY_ONLY 从原网格保留受人体轮廓支持的三角面。DN_GW 从冻结的 U600 状态继续 200 步，输出彩色 RAW、定向 P/N、DPSR Base 和 Refiner Final。2M PLY 保留显示源的彩色原点，并添加带父像素索引的同源插值点。

## 环境与外部输入

从仓库根目录执行。历史实现使用 Windows、Python 3.10、NVIDIA GPU、CUDA 13、MSVC 2022 和 CUDA 架构 12.0。教师训练和首面栅格使用独立的原生 CUDA 环境。学生用 PyTorch，SIFT 准备使用 COLMAP 与 pycolmap。独立 QA 需要 Open3D、h5py、Pillow、OpenCV、NumPy、SciPy、Trimesh 和 plyfile。学生 Final 渲染另需要可工作的 gsplat CUDA 后端，等值面导出需要 scikit-image，独立 SMPL-X 先验栅格需按实际运行路径提供 PyTorch3D。

在 PowerShell 中填写下列路径。结果根承载外部历史输入，输出根承载本次续接目录。为重放归档端点，应将输出根设到相应历史端点的工作副本。单独验证绑定时可用空的临时输出目录。

```powershell
$env:GW40_RESULTS_ROOT   = 'E:/research/gw40-history'
$env:GW40_ARTIFACTS_ROOT = 'E:/research/prepared/thuman20'
$env:GW40_OUTPUT_ROOT    = 'E:/research/replay-output'
$env:GW40_DATASET_ROOT  = 'E:/datasets'
$env:GW40_SMPLX_MODEL   = 'E:/datasets/smplx/SMPLX_NEUTRAL.npz'
$env:GW40_VIT_PYTHON    = 'E:/envs/vggt/python.exe'
$env:GW40_QA_PYTHON     = 'E:/envs/qa/python.exe'
$env:GW40_COLMAP       = 'E:/envs/vggt/Library/bin/colmap.exe'
$env:GW40_VENDOR_ROOT   = 'E:/vendor/GaussianWrapping'
$env:GW40_TEACHER_PYTHON = 'E:/envs/gw-teacher/Scripts/python.exe'
$env:GW40_RASTER_PYTHON  = 'E:/envs/gw-raster/Scripts/python.exe'
```

如编译工具或字体位于其他目录，设置 `GW40_MSVC_VCVARS`、`GW40_CUDA_HOME`、`TORCH_CUDA_ARCH_LIST`、`GW40_QA_FONT` 和 `GW40_MONO_FONT`。`GW40_RAW_SOURCE_ROOT` 可用于配置历史六源样本物料化脚本的 RAW 输入位置。

| 来源 | 需要准备的输入 |
|---|---|
| `GW40_RESULTS_ROOT/DNA_GW_OFFICIAL_DIRECT_20261102` | 教师 `RUN_CONTRACT.json`，0019 控制所引用的居中标定、网格、四相机 BODY 和原始 QA 回执。 |
| `GW40_RESULTS_ROOT/DNA_GW_GEOMETRY_STUDENT_20261103` | `student/TRAINING_CONTRACT.json`、U600 初始化及其 AdamW 和 Python/NumPy/PyTorch/CUDA 随机态、原 `student/DN_GW/DRAW_PLAN.json` 与 200 行 `updates.jsonl`，以及 `evaluation/ENDPOINT18_MANIFEST.json` 和所引用的历史比较输出。 |
| `GW40_RESULTS_ROOT/DNA_0019_MATCHED_REPLAY_GHOST_GATE_20261125` | 冻结规则 `GW40_TARGET_SELECTION_RULE.json`、`FOUR_FRAME_FROZEN_INPUTS.json`、原 0019 对照与逐帧教师 QA。若续用已有的教师端点，应保留完整同帧 `FRAME_INPUT.json`、BODY、教师/QA 端点及准入依据。 |
| `GW40_ARTIFACTS_ROOT` | `DNA_MULTISUBJECT_REPAIR_20261024/inputs/<sample>` 的 `SPLIT40_8.json`、`*READ48_RECEIPT.json` 和登记的同帧 RGB/Mask。学生行还需六视图 `input_npz`、`target_npz`、冻结聚合器 `snapshots`、`lineage`、标定和实际聚合器 RGB。 |
| `GW40_OUTPUT_ROOT/GW40_<frame>/role_inputs` | 诊断帧 0097、0121、0034 的 40/8 角色与 `READ48_RECEIPT.json`。已有原始标定与 SMC 时可用 `python -m research.dna_gw40_dngw0019_replay_20261125.runtime.prospective_role_inputs <frame>` 在工作副本生成对应文件。0097 另使用原 52/8 的同帧输入回执。 |
| 权重与数据 | 固定原 VGGT 底座权重与深度头 FT 检查点、U600、同帧 DNA-Rendering RGB/标注/Kinect SMC、SMPL-X 模型和 THuman2.0 数据。 |

历史 JSON、NPZ 和检查点配置中的路径仍需解析到同一个实际文件，包含 `base_weights`、`ft_checkpoint`、每帧原始资产及冻结特征。配置目录变量后，应保留原路径可访问，或在外部工作副本中按同源事实一致调整路径与回执。U0 比较会检查模型参数、AdamW、随机态和运行配置。

历史教师源码使用 GaussianWrapping 基线 `11e3b6fb5ca6f0a54e2b5587a09693488f3655af` 和 nvdiffrast 基线 `729261dc64c4241ea36efda84fbf532cc8b425b8`。在仓库根目录记录补丁位置后，分别从两个上游源码根目录应用：

```powershell
$patches = (Resolve-Path research/dna_gw40_dngw0019_replay_20261125/patches).Path
Set-Location $env:GW40_VENDOR_ROOT
git apply "$patches/OFFICIAL_DIRECT_SOURCE_ADAPTATION.patch"
Set-Location submodules/nvdiffrast
git apply "$patches/OFFICIAL_NVDIFFRAST_TORCH29_API.patch"
```

补丁目录包含 GaussianWrapping、其 rasterizer 与 nvdiffrast 的原版许可证。按归档环境记录构建 CUDA 自定义扩展。对精确重放，以实际修改与构建记录为准。

原 VGGT 模型与源、GaussianWrapping、SMPL-X 和数据通过各自渠道取得，按上游许可使用。项目的 DPSR 实现附带 [Shape as Points 许可](../mentor_human_model/third_party/SAP_LICENSE.txt)。[VGGT](https://github.com/facebookresearch/vggt) · [GaussianWrapping 许可](https://github.com/diego1401/GaussianWrapping/blob/main/LICENSE.md) · [SMPL-X 模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html) · [DNA-Rendering](https://dna-rendering.github.io/inner-download.html) · [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset)

## 运行阶段

```powershell
python -B -X utf8 research/dna_gw40_dngw0019_replay_20261125/replay.py --help
python -B -X utf8 research/dna_gw40_dngw0019_replay_20261125/replay.py 0008 --until bind
```

| 终点 | 运行内容与主要回执 |
|---|---|
| `bind` | 读取原学生输入、冻结特征、原始角色回执及已存在的教师相机绑定。写入 `continuations/<frame>.json`。诊断帧首次绑定还会读取保存的聚合器输入与原生标定。 |
| `teacher` | 40 TRAIN 路 SIFT、穷举匹配和已知位姿三角化，居中投影，GaussianWrapping 标称 30,000 次迭代、网格提取和 1,000 步纹理优化。用物理相机 02、26、11、35 的原 mask 裁出 BODY，阈值 128，边缘扩张 8 像素，七个三角面采样位置各需三票。比较 8 HELD RGB/Mask 与原生 Kinect 首面，写入 `GW40_RUN.json`、`BODY_ONLY_RECEIPT.json`、`TEACHER_QA_SUMMARY.json`。 |
| `targets` | 按物理六源 02、11、17、26、35、41 的原 K/E 栅格，写入冻结深度 V 和有向伪法线 VN。V 使用原 Kinect 有效支持并要求教师首面有效。VN 还要求两次首面深度一致与邻域连续。部分诊断帧在 `teacher` 终点已完成冻结。 |
| `student` | 先检查首批法线头梯度与实际 U0，后执行 200 步、800 个样本槽。每步只替换首 DNA 槽，其他一个 DNA、两个 TH 槽保持原抽样。新增 GW 深度/法线权重分别为 1.0/0.05。写入 `DN_GW/checkpoint_u0200.pt`、`updates.jsonl` 与逐步读回。 |
| `evaluate` | 用本帧端点复算本帧与固定 18 例。写 `source_RAW.npz`（六源 RAW 与定向 P/N）、完整前景彩色 `source_RAW_all_foreground.ply`、六源 `render.npz`、`base_mesh.ply`、`clothed_mesh.ply` 和表面/高斯回执。表面导出失败时保留 RAW 及失败阶段。 |
| `display` | 默认终点。读取本帧彩色 RAW，对同一源的连续 2×2 像素格插值。输出 2,000,000 点 PLY、源点与父格索引及 `display_2M/<sample>/RECEIPT.json`。 |

已登记的 0019、0008、0047、0102、0097、0121、0034 使用同一入口。0019 只读回归档控制数组和训练抽样。其余帧沿实际回执续接。已完成教师分支从历史目录读取其已有输入，缺阶段的完整重放应在工作副本上执行。首面目标、网格和逐帧准入必须绑定同一物理帧。独立使用六视图 evaluator 时可直接提供同协议的冻结回执清单：

```powershell
& $env:GW40_VIT_PYTHON -B -X utf8 -m research.dna_b_adapter_20261021.run_integrated evaluate --checkpoint 'E:/inputs/checkpoint_u0200.pt' --manifest 'E:/inputs/eval_manifest.json' --output 'E:/research/eval-output'
```

## 验证与局限

- 在 Windows 隔离检出完成 Python 语法、CLI 和各阶段模块导入检查。使用现成历史输入执行 `0008 --until bind`，得到 `BOUND_FRAME`。科研资产下载、CUDA 内核执行、权重加载以及新的教师、学生与评价前向未在隔离检出执行。
- 某些历史端点使用曾经参与旧训练的相机或样本。0097 本轮 HELD 里的 00、06 参加过旧教师训练。0034 的本帧端点评价在该诊断分支内属于训练内结果。
- 历史分支有完整 RAW、Base、Final 和 2M 输出。0097、0121、0034 的整身 RAW 仍有鬼影，教师裁切和插值后的显示密度无法解决这些几何问题。
