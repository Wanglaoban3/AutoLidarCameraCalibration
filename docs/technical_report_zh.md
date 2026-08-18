# LiDAR-Camera 大角度外参自动校准技术报告

## 1. 问题与目标

车辆具备 1 个 LiDAR、6 个相机、IMU 与 GNSS。人工外参可作为初值，但路测中可能产生机械振动、安装偏移或维护后的外参漂移。本报告的目标是恢复共同的 LiDAR-to-body 修正量，优先覆盖 `4 deg` 以内的 roll、pitch、yaw 偏差，并由六相机图像验证结果。

系统分成两层：

1. **大角度粗初始化**：真实 LiDAR 扫描 ICP 形成相对运动，结合 GNSS/IMU 车体相对运动解手眼标定。这一层对图像纹理不敏感，负责姿态进入正确收敛盆地。
2. **图像几何精修与验证**：移除动态目标后，投影 LiDAR 深度不连续边缘，与 Canny/LSD 图像边缘做方向门控的点到线匹配。它适合小漂移精修、质量评分和可视化，不单独承担大 yaw 搜索。

## 2. 一帧真实输入

实验中的一条样本由 nuScenes `sample` 记录给出。处理一帧时需要：

| 输入 | 频率/来源 | 用途 |
|---|---|---|
| `LIDAR_TOP` `.bin`，每点 `[x,y,z,intensity,ring]` | LiDAR | ICP、地面平面、深度不连续边缘 |
| 六张 `CAM_*` 图像及内参 | 25 Hz 相机 | 图像边缘、投影验证 |
| LiDAR 与相机时间戳 | 传感器元数据 | 时间对齐与运动补偿 |
| `ego_pose` | IMU/GNSS 融合输出 | 车体相对运动 `B_k` |
| 手工初值 `T_B_L_manual` | 标定文件 | ICP 初值和待校正外参 |
| 3D 动态目标框，可选 | 感知输出 | 去除车辆、行人、可动物体点 |

本仓库不再分发 nuScenes 原始图像和点云。使用已授权数据时，下列命令会从一帧真实输入生成相机图像、Canny/LSD 边缘、完整点云投影和 LiDAR 深度边缘图：

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/nuscenes_edge_demo.py \
  --dataroot /data --frames 1 --out /workspace/results/one_frame
```

例如前视图为 `results/one_frame/CAM_FRONT_full_lidar_initial.jpg`，对应的 LiDAR 边缘图为 `CAM_FRONT_lidar_edges_initial.jpg`。彩色投影点按深度着色；红点是由局部投影邻域深度跳变选出的 LiDAR 遮挡边缘。

## 3. 坐标系与统一状态

采用列向量变换，`T_A_B` 将坐标系 `B` 中的点变换到 `A`。`B` 表示车体坐标系，`L` 表示 LiDAR，`C_i` 表示第 `i` 路相机。

六路相机不应独立发布六份 LiDAR 修正，而应共享一个车体系扰动 `xi_BL`：

```text
T_Ci_L(xi_BL) = T_Ci_B * exp(xi_BL) * T_B_L_manual
```

因此，任何一路图像只是对相同物理量的一个观测。若某一相机长期偏离，只能诊断为该相机自身支架或内参异常，不能把它混入共同 LiDAR 外参。

## 4. 大角度粗初始化：ICP 与手眼标定

相邻两帧 LiDAR 原始点云经过体素下采样、距离裁剪和点到点 ICP，得到 LiDAR 相对运动 `A_k`。由 GNSS/IMU 位姿求车体相对运动 `B_k`。二者满足手眼关系：

```text
B_k * T_B_L = T_B_L * A_k
```

以人工初值的左乘增量表示待估量：

```text
T_B_L = exp(delta_xi) * T_B_L_manual
```

用 Huber 损失最小化所有运动对的 SE(3) 残差：

```text
r_k = Log( inverse(B_k * T_B_L) * (T_B_L * A_k) )
min sum rho(||r_k||^2)
```

仅相邻帧的短基线对 yaw 和平移可观性不足，因此实现将可靠相邻 ICP 运动复合成 1、3、6 帧基线。车辆必须包含转弯、加减速或路面起伏等激励；长期直行会使某些自由度退化。

## 5. Galibr-style 图像验证与精修

### 5.1 图像侧

对灰度图高斯平滑后计算 Canny，并提取 LSD 长线段。图像距离场 `D(u)` 是到最近图像边缘的欧氏距离。LSD 线段为精修提供端点有限的点到线距离和方向。

### 5.2 LiDAR 侧

先使用 3D 框或感知实例掩码去掉车辆、行人和可动物体。去除车体附近地面后投影静态点；在图像平面局部邻域内搜索深度变化，深度跳变大于阈值的点构成遮挡/轮廓边缘。局部 PCA 给每个 LiDAR 边缘点估计切线方向。

### 5.3 匹配与优化

对投影点 `u_j` 与候选 LSD 线段 `l_j`，只保留距离和无向切线夹角同时满足门限的候选。细优化使用：

```text
min sum rho( point_to_segment(u(T_Ci_L * p_L), l_j) ) + prior(delta_xi)
```

粗阶段不得固定错误对应关系。正确顺序是：手眼或地面约束得到姿态盆地，重新建立边缘对应，再做连续优化。每轮后使用帧中位残差和 MAD 剔除不一致帧，并在留出帧上评分。

## 6. 4 度真实点云模拟实验

数据为 nuScenes Mini 场景 `scene-0061`，使用 20 帧真实原始 LiDAR 扫描。注入的共同车体系旋转向量为 `[3, -3, 4] deg`，并注入平移 `[0.06, -0.04, 0.08] m`。LiDAR 相对位姿由原始点云 ICP 估计；`ego_pose` 模拟 GNSS/IMU。

| 指标 | 数值 |
|---|---:|
| 期望姿态修正 | `[-3.00, 3.00, -4.00] deg` |
| ICP+手眼估计 | `[-2.21, 2.23, -3.81] deg` |
| 三轴姿态误差 | `[0.79, 0.77, 0.19] deg` |
| 相邻 ICP 中位 RMSE | `0.547 m` |
| 使用的复合运动对 | 50 |

结果表明，在该真实点云模拟中，4 度以内共同姿态偏差已被恢复到约 1 度以内。平移尤其是竖直方向仍不可可靠恢复，原因是车辆大部分时间在近似平面上运动，手眼问题对该自由度退化。

可复现命令：

```bash
docker build -t auto-lidar-camera-calibration .
docker run --rm --entrypoint python \
  -v "$PWD:/data:ro" -v "$PWD/results:/workspace/results" \
  auto-lidar-camera-calibration /workspace/lidar_icp_handeye.py \
  --dataroot /data --frames 20 --noise-rpy-deg 3 -3 4 \
  --out /workspace/results/icp_handeye_4deg
```

输出报告中的 `expected_body_correction` 只用于实验评价；真实车端没有这项真值，应改由独立图像验证窗口、安全门和长期稳定性判定来决定是否发布。

## 7. 发布安全门与量产建议

- 至少三路相机、多个时间分散窗口支持同一车体系修正。
- 留出帧静态边缘残差优于人工外参，且优于第二候选有足够裕量。
- 估计值不位于粗搜索边界；逐相机留一验证保持一致。
- 估计时钟偏移，避免将传感器不同步误吸收到 yaw 或平移中。
- 仅在车辆存在足够转弯/激励时运行手眼更新；退化窗口冻结外参。
- 发布后使用低通、速率限制和可回滚版本管理。

生产实现建议用 KISS-ICP、FAST-LIO 或同等级 LiDAR 里程计替换本仓库为可复现而写的简化点到点 ICP；图像侧建议替换通用 Canny 纹理边缘为道路边界、路缘、杆、立面等静态语义边缘。

## 9. 为什么结果会漂移

当前原型的手眼阶段确实恢复的是 `T_B_L`，但它不是一个独立的“真值恢复器”。现有 ICP 使用：

```text
A_seed = inverse(T_B_L_manual) * B_k * T_B_L_manual
```

因此人工外参会影响扫描配准的收敛盆地；随后手眼又把 ICP 的 `A_k` 当作观测。ICP 的局部错误就会被手眼优化解释成 LiDAR 外参变化。车辆近似平面运动还会让 z 平移和部分姿态自由度退化。

仓库现在提供两个诊断选项：

```bash
# 使用人工外参和 IMU 运动作 ICP 初值（当前基线）
... lidar_icp_handeye.py --icp-seed manual

# ICP 从单位变换开始，显式观察不依赖人工外参时的漂移
... lidar_icp_handeye.py --icp-seed identity
```

`report.json` 中的 `gap_estimates` 会分别报告 1、3、6 帧运动基线的手眼结果。如果这些结果在旋转或平移上不一致，不能发布单个全局外参。一次实际诊断中，`manual` seed 的 ICP 中位 RMSE 为 `0.547 m`，`identity` seed 为 `0.800 m`；后者的手眼 RMSE 为 `4.36`，说明这个简化 ICP 对初值敏感。

## 10. 消除初值依赖的下一版

量产实现应把外参、每帧 LiDAR 位姿和传感器时间偏移放进同一个因子图，而不是串联地把 ICP 结果当真值：

1. LiDAR odometry 在 LiDAR 自身坐标系中运行，只使用 IMU 陀螺作为旋转先验，不使用待校正的 `T_B_L` 作为几何约束。
2. 对每一段运动保存 ICP/scan-matching 协方差和退化指标；平面直行、动态物体过多或重叠不足的运动不进入手眼因子。
3. 用多个独立时间窗口、留一窗口和 RANSAC/Huber 估计 `T_B_L`，计算窗口间协方差；漂移超过阈值时冻结而不是平均发布。
4. 将六相机的静态语义边缘、地面法向和杆/立面线加入同一个全局目标，用图像因子打破纯车辆运动下的 yaw/平移歧义。
5. 对至少两个粗姿态假设并行精修，在留出帧上比较得分；粗匹配不应成为唯一初始化路径。

因此，手眼标定应被视为“粗初始化和运动约束因子”，而不是最终外参。最终发布量必须由 LiDAR、IMU/GNSS、六相机和时间偏移的联合一致性决定。

## 8. 文件说明

| 文件 | 作用 |
|---|---|
| `lidar_icp_handeye.py` | 真实 LiDAR 点云 ICP + GNSS/IMU 手眼标定模拟 |
| `handeye_initializer.py` | 使用 oracle LiDAR 位姿验证手眼数学链路，非生产输入 |
| `joint_body_calibration.py` | 六相机共享车体系修正、地面初始化与边缘验证 |
| `nuscenes_edge_demo.py` | 单帧/多帧 Galibr-style 边缘投影、动态过滤和可视化 |
| `results/*.json` | 已完成实验的数值输出 |
