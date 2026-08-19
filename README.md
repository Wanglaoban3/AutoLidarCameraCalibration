# Auto LiDAR-Camera Calibration

面向量产车 LiDAR-camera 外参漂移的可复现实验工程。目标是从质量一般的人工初始外参出发，使用 LiDAR、六路相机、IMU/GNSS 恢复共同的 LiDAR-to-body 姿态修正，再通过图像边缘完成验证与小范围精修。

当前实验使用 nuScenes Mini 的真实 `LIDAR_TOP` 和六路相机帧；数据集本身不包含在仓库中。

## What Works

- 真实原始 LiDAR 扫描的 ICP 相对运动估计。
- GNSS/IMU 车体轨迹与 LiDAR ICP 的手眼标定。
- 注入 `roll=3 deg, pitch=-3 deg, yaw=4 deg` 车体系扰动时，姿态恢复为 `[-2.21, 2.23, -3.81] deg`。
- Canny + LSD + LiDAR 深度不连续边缘的 Galibr-style 图像验证与可视化。
- nuScenes 3D 标注的动态对象过滤、六相机投影和发布安全门。

## Quick Start

准备已授权的 nuScenes Mini 数据集，使仓库根目录下有 `v1.0-mini/`、`samples/` 和 `sweeps/`。数据文件已被 `.dockerignore` 排除，不会被提交。

```bash
docker build -t auto-lidar-camera-calibration .

docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/lidar_icp_handeye.py \
  --dataroot /data --frames 20 --noise-rpy-deg 3 -3 4 \
  --out /workspace/results/icp_handeye_4deg
```

比较 ICP 对人工初值的依赖：把 `--icp-seed manual` 改为 `--icp-seed identity`。报告中的 `gap_estimates` 会分别输出 1、3、6 帧基线的手眼结果；窗口间明显不一致时应冻结校准。

## Inspect Edge Matches

固定 ICP+手眼粗标结果，生成三联图：图像 Canny/LSD 线段、LiDAR 候选边缘点、以及最终接受的点到线匹配。

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/edge_match_inspection.py \
  --dataroot /data --frames 3 \
  --coarse-json /workspace/results/icp_handeye_diagnostic_manual_4deg.json \
  --out /workspace/results/edge_match_inspection
```

每路相机输出 `*_matches.jpg`、`*_image_edges.jpg`、`*_lidar_candidates.jpg`、`*_accepted_matches.jpg` 和 `report.json`。三联图中红点是所有投影 LiDAR 边缘候选，绿点是通过距离/方向门控的匹配，紫线是点到图像线段的残差连接。

## Inspect Historical Stacking And Ground Returns

以窗口最后一帧为参考，使用真实 ego pose 将此前 LiDAR 扫描补偿并堆叠到参考相机。每个堆叠尺度都会输出全点、地面回波/高反射候选、深度不连续边缘以及三联对比图。

```bash
docker build -t auto-lidar-camera-calibration .

docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/stacked_lidar_demo.py \
  --dataroot /data --scene 0 --frames 10 --stack-counts 1 5 10 \
  --camera CAM_FRONT --out /workspace/results/stacked_lidar_front
```

查看 `CAM_FRONT_history_01_comparison.jpg`、`CAM_FRONT_history_05_comparison.jpg` 和 `CAM_FRONT_history_10_comparison.jpg`。黄色点是地面平面内的静态 LiDAR 回波，青色点是其中强度前 10% 的候选。后者只能帮助检查车道线是否可能被采到，不能作为车道线语义标签或直接用于外参优化；真正使用前仍需在图像侧做车道线/路缘语义分割，并以地面法向和时序一致性门控。

上面的 `--frames` 读取相隔约 0.5 秒的关键扫描，只适合长时间结构检查。要获得短时间地面稠密化，加入 `--sweeps 20 --stack-counts 1 5 10 20`；它从参考扫描沿 `sample_data.prev` 读取约 20 Hz 的连续历史 sweep。sweep 没有逐帧 3D 标注，动态点以时间最近的 2 Hz 标注近似过滤，因此只应将其用于静态道路结构的候选检查。

## Inspect TEED And Stacked Ground Returns

TEED 是轻量深度边缘模型，不提供箭头语义标签。此检查将 10 个连续 sweep 的高反射地面候选与 TEED 边缘概率相交，输出可用于后续标定的结构边缘候选。镜像在构建时固定下载官方 BIPED epoch-5 权重并校验 SHA-256。默认镜像使用 CPU PyTorch；TEED 只有 58K 参数，足以用于该离线检查。

```bash
docker build -t auto-lidar-camera-calibration .

docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/teed_ground_inspection.py \
  --dataroot /data --scene 0 --reference-offset 9 --sweeps 10 --teed-percentile 95 \
  --camera CAM_FRONT --out /workspace/results/teed_ground_front
```

输出 `CAM_FRONT_comparison.jpg`：依次为 TEED 边缘、高反射候选、BEV 栅格化后的 LiDAR 强度轮廓和 TEED 融合结果。默认以当前图像的 TEED 概率 95 分位自适应选取强边缘，实际阈值记录在 `report.json`；可用 `--teed-threshold` 进行固定阈值对照。对低阈值地面标识可使用 `--sweeps 20 --intensity-percentile 70`，再由 `CAM_FRONT_intensity_contours.jpg` 检查箭头/车道线是否形成连续边界。轮廓不带箭头语义，仍须以留出帧验证时间稳定性。

## Recover A Noisy Extrinsic With TEED

先生成真实原始 LiDAR ICP + 手眼粗标，再以连续 sweep 的 BEV 强度轮廓和六相机 TEED 距离场做共享车体系姿态精修。训练和留出关键帧分开；默认利用地面法向约束横滚/俯仰并冻结平移，报告只在留出帧距离也下降且未触边界时将 `publish_attitude` 置为 true。

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/lidar_icp_handeye.py \
  --dataroot /data --scene 0 --frames 20 --noise-rpy-deg 3 -3 4 \
  --out /workspace/results/icp_teed_coarse

docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/teed_stacked_refinement.py \
  --dataroot /data --scene 0 --coarse-json /workspace/results/icp_teed_coarse/report.json \
  --cache-dir /workspace/results/teed_cache_scene_0 \
  --train-offsets 4 6 8 --holdout-offsets 10 12 --sweeps 20 --intensity-percentile 70 \
  --out /workspace/results/icp_teed_refined
```

查看 `report.json` 的 `coarse_error_rpy_deg`、`refined_error_rpy_deg`、`train_score_*` 和 `holdout_score_*`。这是基于 nuScenes ego pose 的短时 sweep 运动补偿评估；生产系统必须用 LiDAR odometry/ICP 轨迹替换该补偿链路。

地面标识边缘对平移的可观测性弱，默认冻结平移。只有增加独立的建筑/路缘约束后，才应以 `--refine-translation` 开启受限平移更新；任一优化自由度触边界时，发布标记会被强制置为 false。

## Cache TEED And Refine Roll From Vertical Contours

TEED 推理只需对每张相机图执行一次。先为场景的全部关键帧和六路相机预计算概率图；缓存以 `sample_data` token 命名，后续 roll 实验直接读取 `.npy`，不会再次推理。竖直精修在每个参考时刻叠加 20 个连续 sweep，使用非地面点的 z-buffer 深度突变和图像竖直连续性提取杆状/树干/建筑竖边候选；以地面精修结果为初值，仅更新 roll。

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/teed_edge_cache.py \
  --dataroot /data --scene 0 --cache-dir /workspace/results/teed_cache_scene_0

docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/teed_vertical_roll_refinement.py \
  --dataroot /data \
  --initial-json /workspace/results/icp_teed_refined_contours_all_cameras/report.json \
  --cache-dir /workspace/results/teed_cache_scene_0 \
  --train-offsets 4 6 8 --holdout-offsets 10 12 --sweeps 20 \
  --out /workspace/results/teed_vertical_roll
```

`report.json` 中的 `cache.inferred` 必须为 `0` 才说明本次优化完全使用缓存；检查 `train_refined.jpg` 和 `holdout_refined.jpg` 中绿色 LiDAR 点是否落在橙色竖直 TEED 轮廓附近。滑动窗口应共享同一个车体系外参并在多个参考时刻上联合优化，不能把所有历史点直接堆到一个很长的窗口中。

查看 `results/icp_handeye_4deg/report.json` 中的 `expected_body_correction` 与 `estimated_body_correction`。完整方法、单帧输入和限制见 [技术报告](docs/technical_report_zh.md)。

## Important Limitation

ICP + 平面车辆运动可可靠恢复姿态，但竖直平移通常不可观。该工程是研究原型，不应直接用于车端发布；生产系统还必须接入时钟偏移估计、地图/语义静态物体、长时间激励轨迹、独立验证窗口和速率限制。

## Python MFCalib

`mfcalib_python.py` 是 MFCalib 风格核心流程的无 ROS 版本，使用 NumPy、SciPy、OpenCV 和 PyYAML。它读取一张图像、一个点云和一个 YAML 配置，支持 `.npy`、nuScenes 风格 `.bin`（每点 3/4/5 个 float）以及 ASCII/float32 `.pcd`。

```bash
docker build -f Dockerfile.mfcalib -t mfcalib-python .
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  mfcalib-python /workspace/mfcalib_python.py \
  --image /data/example/image.jpg \
  --points /data/example/points.npy \
  --config /workspace/mfcalib_config.example.yaml \
  --out /workspace/results/mfcalib_python
```

输出 `report.json`、`image_edges.png`、`lidar_sphere_edges.png` 和 `mfcalib_overlay.jpg`。ROS bag 读取被文件输入替换；算法阶段对应官方单帧 MFCalib 流程，数值后端使用 NumPy/SciPy 替代 PCL/Ceres。

### nuScenes-mini Segment Benchmark

`mfcalib_nuscenes.py` 将 nuScenes-mini 的一段连续 `LIDAR_TOP` 数据补偿并堆叠到窗口最后一个 LiDAR 帧，再用同一时刻的指定相机帧运行 MFCalib。`calibrated_sensor` 和 `ego_pose` 只用于注入可复现的外参扰动、窗口运动补偿和最终误差评估；报告会明确标记这是 oracle-trajectory benchmark。

```bash
docker build -f Dockerfile.mfcalib -t mfcalib-python .
docker run --rm --entrypoint python \
  -v /home/wangyi/projects/galibr_lab:/data:ro \
  -v "$PWD/results:/workspace/results" \
  mfcalib-python /workspace/mfcalib_nuscenes.py \
  --dataroot /data --scene 0 --start 0 --frames 8 --camera CAM_FRONT \
  --noise-mode uniform --seed 7 \
  --out /workspace/results/mfcalib_nuscenes_scene0_front_seed7
```

`--noise-mode uniform --seed N` 在每一维独立采样，范围为 `±[3°, 3°, 4°]` 和 `±[0.06, 0.04, 0.08] m`；`--noise-mode fixed` 则使用参数给出的带符号扰动。检查输出根目录的 `initial_error_rpy_deg`、`final_error_rpy_deg`、`sampled_noise_se3`、`accepted_stages`。若所有阶段都拒绝更新，表示该窗口在当前特征/初值下不可观，工具安全保留初值。
