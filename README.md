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

查看 `results/icp_handeye_4deg/report.json` 中的 `expected_body_correction` 与 `estimated_body_correction`。完整方法、单帧输入和限制见 [技术报告](docs/technical_report_zh.md)。

## Important Limitation

ICP + 平面车辆运动可可靠恢复姿态，但竖直平移通常不可观。该工程是研究原型，不应直接用于车端发布；生产系统还必须接入时钟偏移估计、地图/语义静态物体、长时间激励轨迹、独立验证窗口和速率限制。

