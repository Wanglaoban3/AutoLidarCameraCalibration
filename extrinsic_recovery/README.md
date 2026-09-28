# 相机 / LiDAR 外参恢复（pitch & roll）

以 nuScenes 官方标定为 GT，对 CAM_FRONT 与 LIDAR_TOP 的外参旋转注入固定
mounting 噪声（仅 pitch/roll，yaw 视为 IMU 已知量保持不变），再用场景特征
将其恢复，评估恢复残差相对注入噪声的下降。

## 运行方式

```bat
H:\miniconda3\envs\yt\python.exe camera_vp_calib.py     > cam_out.txt   2>&1
H:\miniconda3\envs\yt\python.exe lidar_gravity_calib.py > lidar_out.txt 2>&1
```

依赖：yt 环境的 torch / ultralytics(yolo26x-sem.pt) / opencv(LSD) / scipy，
TEED 权重在 `H:\projects\TEED-main\checkpoints\BIPED\7\7_model.pth`，
数据 `H:\datasets\nuscenes-mini` + `H:\datasets\can_bus_extract\can_bus`。

## 实验结果（噪声 ±2° 均匀分布，SEED 固定）

| 传感器 | 注入噪声 | 恢复后残差 | 降幅 |
|---|---|---|---|
| CAM_FRONT（VP 竖线法 + 梯度亚像素精修 + 逆方差聚合） | 1.666° | **0.507°** | 70% |
| LIDAR_TOP（地面锚点 + 立面正交约束） | 1.486° | **0.364°** | 76% |

## 相机管线（camera_vp_calib.py）

1. 帧集：3 个场景 log（Boston 街区 + 新加坡高层 x2）共 30 个 CAM_FRONT 关键帧。
2. 噪声注入（仅一次，模拟固定安装偏差）：`R_noisy = Cn @ R_gt`，
   `Cn = Rz(0) Ry(ep) Rx(er)`（ego 系）。
3. 特征：YOLO-seg 抠建筑 → TEED 原生分辨率概率图（min-max 归一化！）→
   LSD 直线段（限建筑内近竖直）→ **梯度亚像素定位**（Sobel 幅值垂直方向
   峰值 + 二次插值，TEED 概率图做验证门）→ 加权 TLS 拟合线方向。
   线内角度一致性 std 0.01–0.12°（逐帧打印 line ang std）。
4. 竖直方向估计：方向空间最小特征向量（**不求 VP 交点**——前视相机竖直线
   VP 在 14.6 万像素外的无穷远，有限 VP 求交完全病态），RANSAC + IRLS，
   带"消失方向必须近竖直"门控（防水平线族错锁）。
5. 逐帧重力目标：ego 姿态旋转把全局 up 转到 ego 系（坡道/悬挂补偿）。
   CAN-IMU 加速度计已接入比对（均值差 5.55°——行驶动力学污染，仅作
   交叉验证诊断，不参与恢复）。
6. 修正量 C2 = 对齐(u → 重力) 去掉 yaw 分量；多帧聚合：中位数 2° 内的
   一致性筛选 + 测量信息量（Σ 内线长度×TEED 权重）逆方差加权四元数平均。
7. 水平线第二 VP（地平线拟合 → v = Kᵀ l_h）带 5° 一致性门控，仅在与
   竖线估计一致时融合。

## LiDAR 管线（lidar_gravity_calib.py）

1. 12 帧 sweep，每帧多平面 RANSAC（20 平面 / 500 迭代 / 3 cm / SVD 精化）。
2. 近水平平面（|n_z|>0.9）中内点最多者为地面 → **4 m 网格逐格 SVD 法向、
   加权平均**作为竖直轴绝对锚点（消除单平面坡度/起伏偏差）。
3. 近竖直平面（|n_z|<0.25，建筑立面）为正交约束（弱权重 0.05——本场景
   老砖楼立面实测自带 ~1° 真实倾斜，强约束会带偏解）。
4. 逐帧 IMU 重力比对：立面法向解出的逐帧重力 vs 该帧 ego 姿态重力，
   中位差 2.23°；差 >8° 的 sweep 标记不可靠（法向简并）。
5. 优化参数限制在垂直于重力轴的切平面内（2 参数）——绕重力轴旋转不可观
   测且不属于 pitch/roll，防止优化器漂移。

## 关键教训（踩过的坑）

- 欧拉角对比在相机安装奇异性（roll≈-89.5°）附近失真，指标必须用
  修正量与注入量的测地距离。
- 前 viewport 竖线 VP 在无穷远，任何"有限 VP 求交"方案都病态。
- 特征向量有 ±符号二义性，对齐前必须定向（否则解出 180° 翻转）。
- 立面法向简并（平行立面只约束 1 个自由度）时必须引入先验或地面锚点。
- BEV 地面对齐做优化目标会被地面不平 + 虚线周期性带进假解（实测滑到
  9° 假解）；极线/平面正交约束才是良构目标，BEV 只做验证。
- nuScenes 图像已去畸变（K 即无畸变内参），畸变不在怀疑列表内。
- ego 姿态逐帧变化（坡道俯仰 0.5~1.2°）必须逐帧补偿，固定 [0,0,1]
  当重力会把坡度当安装误差。

## 产物

- `results/camera_R_ec_recovered.npy`、`results/lidar_R_le_recovered.npy`
  恢复后的外参旋转。
- `results/camera_vp_frame{0,1}.png` 建筑竖线共识线段可视化；
  `results/lidar_vertical_planes_bev.png` 立面平面 BEV 着色。
- `cam_out.txt` / `lidar_out.txt` 逐帧明细日志。
