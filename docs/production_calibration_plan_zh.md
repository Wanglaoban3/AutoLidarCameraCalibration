# 面向量产车的 LiDAR-Camera 自动标定技术方案

## 1. 目标与边界

本方案针对以下车辆配置和验收要求：

- 1 个 LiDAR、多个相机、IMU、GNSS 和车辆 Odom；
- LiDAR 和相机相对人工初值都可能存在约 `3 deg` 姿态误差和 `0.10 m` 量级平移误差；
- 量产验收主要关注旋转误差 `<= 0.2 deg`；
- 平移误差不作为最终验收指标，但不能被错误地吸收到 roll/pitch/yaw；
- 最终业务 KPI 是 BEV 视角下的车道线对齐，而不是单纯的像素边缘残差。

本方案建议把系统拆成两层：

1. **基准标定层**：产线或维修场使用受控数据得到可信的 `T_B_L`、`T_B_Ci`、相机内参/畸变和时间同步参数。
2. **道路运行层**：使用 IMU、GNSS、Odom、LiDAR odometry 和车道线 BEV 观测监控漂移，只对可观、可信的参数做有限更新。

车上算法不应在每次运行时自由优化全部 6DoF 外参。对于当前目标，发布状态优先只包含公共 LiDAR roll 修正和必要的相机自身 roll 诊断量。

## 2. 推荐系统架构

```text
传感器时间同步与质量检查
        |
        v
IMU预积分 + GNSS/INS + Odom约束
        |
        +--> 车体连续时间轨迹 T_B(t)、重力方向、速度/横摆率
        |
LiDAR scan matching / LiDAR odometry（不依赖待标外参）
        |
        +--> LiDAR相对运动 A_k、协方差、退化指标
        |
GNSS/IMU/Odom - LiDAR hand-eye 粗初始化
        |
        +--> LiDAR-to-body 粗姿态，进入正确收敛盆地
        |
相机车道语义 + LiDAR地面/反射强度 BEV
        |
        +--> 车道虚拟靶标、signed distance、宽度/方向因子
        |
roll-only BEV精修 + 多窗口一致性 + 协方差门控
        |
        +--> 发布有限外参修正，或保持产线基准并上报告警
```

## 3. 坐标系和待估状态

使用列向量变换，`T_A_B` 表示将坐标系 `B` 中的点变换到坐标系 `A`。

- `B`：车体坐标系；
- `L`：LiDAR 坐标系；
- `C_i`：第 `i` 路相机坐标系；
- `G`：GNSS/地图全局坐标系。

产线基准为：

```text
T_B_L^0
T_B_Ci^0
K_i, distortion_i
Delta_t_LC_i^0
```

道路运行时建议使用以下状态：

```text
delta_xi_BL       公共LiDAR-to-body小修正，主发布状态
delta_xi_BCi      相机自身小修正，仅在有证据时诊断/发布
T_B(t)            车体连续时间轨迹
delta_t_LC_i      LiDAR-camera时间偏移
eta_k             第k个短窗口的道路横坡/车身姿态nuisance量
b_xy_k            第k个窗口的BEV二维平移nuisance量
b_g, b_a          IMU bias或其慢变残差
```

如果相机和 LiDAR 都可能有 `3 deg/0.10 m` 初始误差，不能默认相机绝对正确。应使用强产线先验和多路相机一致性，把公共 LiDAR 误差与单相机支架误差分开：

```text
T_B_L = Exp(delta_xi_BL) T_B_L^0
T_B_Ci = Exp(delta_xi_BCi) T_B_Ci^0
T_Ci_L = T_Ci_B T_B_L
```

短时间窗口内不应同时无约束地放开所有 `delta_xi_BCi`。先估计公共 LiDAR 修正，再用各相机残差诊断单相机异常。

## 4. 3 度、10 厘米初始误差的粗初始化

### 4.1 LiDAR odometry 必须独立于待校外参

LiDAR odometry 只在 LiDAR 自身坐标系中运行，输入是连续点云和 IMU 角速度先验，不使用 `T_B_L^0` 将 GNSS/IMU 轨迹变成 ICP 初值。

推荐接口：

```text
lidar_odom(scan_k, scan_k+1, imu_segment)
    -> A_k, covariance_k, degeneracy_k
```

生产后端可使用 KISS-ICP、FAST-LIO、LIO-SAM 或现有量产 LiDAR odometry。当前仓库的简化点到点 ICP 只能作为回归测试，不应直接作为车端 odometry。

### 4.2 GNSS、IMU、Odom 的分工

| 输入 | 主要作用 | 注意事项 |
|---|---|---|
| IMU陀螺/加速度 | 高频姿态、重力、短时连续轨迹 | 必须做 bias 估计和温度补偿 |
| GNSS/INS | 低频全局位置、长期航向和速度约束 | 低速时航向不可靠；双天线或高速行驶时才使用航向因子 |
| 车辆Odom | 纵向速度、非完整约束、短时运动先验 | 需要轮速打滑检测；不要把轮速当作无偏真值 |
| LiDAR odometry | LiDAR坐标系相对运动 | 必须保存协方差、重叠率、退化方向 |

车辆 Odom 应加入非完整约束，例如车体横向速度接近零；急转弯、打滑、低附着路面时降低权重或剔除。

### 4.3 Hand-eye 粗标定

LiDAR 相对运动 `A_k` 和车体相对运动 `B_k` 满足：

```text
B_k T_B_L = T_B_L A_k
```

使用多个非重叠时间窗口、1/3/6 帧复合基线和 Huber/Cauchy 损失估计 `T_B_L`。对每个运动保存权重：

```text
w_k = f(overlap, ICP covariance, rotation excitation,
        translation excitation, dynamic ratio, Odom quality)
```

当初始误差达到 `3 deg` 时，粗初始化至少保留两个或三个姿态候选。若 GNSS/IMU/Odom 与 LiDAR odometry 的 hand-eye 结果在不同窗口不一致，不应平均发布，而应冻结并报告数据退化。

粗初始化搜索建议：

- roll/pitch：优先由 IMU 重力和 LiDAR 地面法向约束；
- yaw：使用车辆转弯、GNSS/INS 航向、Odom 方向和 LiDAR odometry 联合约束；
- 平移：可在优化中作为 nuisance 变量估计，但不作为发布 KPI；
- 相对初始误差可能达到 6 deg 时，图像侧不要从单一人工匹配开始，必须先做距离场/BEV 粗搜索。

## 5. Roll 的核心算法

### 5.1 重力-地面法向因子

在 LiDAR 坐标系拟合地面法向 `n_L`，在车体坐标系使用滤波后的 IMU 重力 `g_B`：

```text
r_gravity = Log_SO3( R_B_L n_L, g_B )
```

工程实现中可以使用叉积形式：

```text
r_gravity = (R_B_L n_L) x g_B
```

要求：

1. 每个短窗口独立使用 RANSAC/加权平面拟合；
2. 只使用车辆前方和近车区域的稳定道路点；
3. 输出平面 RMS、法向协方差和有效点比例；
4. 用 IMU 重力定义车体 roll，不能把道路横坡直接误认为传感器 roll；
5. 对悬架运动使用低通重力和窗口中位数。

当前项目中通过 Rodrigues 旋转向量前两个分量近似 roll/pitch 的做法，应替换为严格的旋转残差和单轴参数化。

### 5.2 车道线作为虚拟标定靶

相机侧使用车道线语义分割或实例折线，LiDAR 侧使用地面点的 ring-normalized 强度、距离归一化强度和多 sweep 静态一致性生成车道候选。

两侧都生成 BEV 表示：

```text
P_cam_bev(x, y)       相机车道概率
P_lidar_bev(x, y)     LiDAR车道候选概率
D_cam_bev(x, y)       相机signed distance field
D_lidar_bev(x, y)     LiDAR signed distance field
```

车道精修目标：

```text
J_lane =
  lambda_1 * Chamfer(D_cam_bev, lidar_points)
  + lambda_2 * Chamfer(D_lidar_bev, camera_lane_points)
  + lambda_3 * lane_direction_error
  + lambda_4 * lane_width_error
  + lambda_5 * temporal_consistency
```

推荐只优化一个 `delta_roll`：

```text
delta_roll in [-0.8 deg, +0.8 deg]
coarse grid: 0.01 deg
fine search: Brent/golden-section
```

窗口内允许估计不发布的 `b_xy_k`，避免 10 cm 量级平移误差被错误吸收到 roll 中。

### 5.3 车道宽度、平行性和实例连续性

车道宽度是 roll 的强辅助约束，但不能作为硬约束：

```text
r_width = observed_width - soft_prior_width
r_parallel = direction_left - direction_right
```

只在左右车道实例都稳定存在时提高权重；匝道、汇入、分叉、施工区和车道线磨损严重时降低权重。

车道候选应具有：

- 多 sweep 连续性；
- 相邻窗口位置和曲率连续；
- 相机语义支持；
- LiDAR 强度/反射一致性；
- 动态目标和阴影排除。

全局 P70/P90 强度阈值只能作为候选生成，不能作为车道语义标签。

## 6. 统一残差和优化流程

推荐的窗口级目标为：

```text
J = J_imu_preintegration
  + J_gnss
  + J_odom_nonholonomic
  + J_lidar_odom
  + J_handeye
  + J_gravity_plane
  + J_lane_bev
  + J_camera_consensus
  + J_time_offset
  + J_factory_prior
```

其中：

- `J_imu_preintegration`：IMU 轨迹连续性和 bias；
- `J_gnss`：位置、速度、必要时航向；
- `J_odom_nonholonomic`：纵向速度和横向速度约束；
- `J_lidar_odom`：LiDAR 自身连续运动；
- `J_handeye`：LiDAR-to-body 粗姿态；
- `J_gravity_plane`：roll/pitch；
- `J_lane_bev`：最终 BEV 车道 KPI；
- `J_camera_consensus`：六路相机共同 LiDAR 修正一致性；
- `J_time_offset`：时间偏移和运动补偿；
- `J_factory_prior`：防止在线估计漂移到错误局部极小值。

推荐采用两阶段优化：

### 阶段 A：粗初始化

1. 时间戳检查和初始 `delta_t` 搜索；
2. IMU/GNSS/Odom 连续时间轨迹；
3. LiDAR odometry；
4. hand-eye 求 `T_B_L` 粗修正；
5. 相机车道消失点/地面轮廓给出相机 pitch/yaw 初值；
6. 使用重力-地面法向得到 roll 初值。

### 阶段 B：有限维精修

1. 固定或强约束 pitch/yaw；
2. 只开放 `delta_roll`；
3. 同时估计窗口内 `b_xy_k` 和道路横坡 nuisance 量；
4. 使用相机车道语义与 LiDAR BEV 车道候选做 signed-distance 优化；
5. 多相机、多窗口联合评分；
6. 计算 Hessian 曲率、协方差和 session 一致性；
7. 满足发布门后才更新。

## 7. 可观性与发布门

每个候选窗口必须记录：

```text
ground_plane_rms
ground_normal_covariance
lane_instance_count
lane_length_m
lane_direction_spread
LiDAR-camera overlap
ICP covariance / degeneracy
roll objective curvature
sigma_roll
session-to-session roll std
```

拒绝条件包括：

- 地面拟合残差过大；
- 只有一条或很短的车道线；
- 左右车道无法区分；
- 车道曲率或施工区域导致宽度先验失效；
- LiDAR 动态比例过高；
- 时间偏移不稳定；
- roll 优化触碰边界；
- Hessian 近奇异或 `sigma_roll` 超限；
- 不同 session 的估计方向不一致；
- 留出窗口 BEV 误差没有改善。

建议内部目标设为：

```text
roll P95 <= 0.10 deg
量产放行 P95 <= 0.20 deg
```

最终验收不应只看角度，还应按距离区间测量 BEV 车道横向误差，例如 `0-10 m`、`10-20 m`、`20-40 m`，并统计 P50/P95/P99。

## 8. 对当前仓库的改造建议

### 保留

- `lidar_icp_handeye.py` 的 hand-eye 数学接口和结果报告框架；
- `handeye_initializer.py` 的 oracle 回归测试；
- `joint_body_calibration.py` 的共享车体系状态；
- sweep 堆叠、动态过滤和 holdout 评估思路；
- TEED 缓存和可视化工具。

### 替换或降级

- 将简化点到点 ICP 替换为生产 LiDAR odometry 接口；
- 将 `teed_vertical_roll_refinement.py` 从主 roll 估计器降级为辅助诊断；
- 新增 `roll_gravity_estimator.py`，严格使用地面法向和 IMU 重力；
- 新增 `lane_instance_tracker.py`，输出带实例 ID 的相机车道线；
- 新增 `lidar_lane_bev.py`，做 ring/距离归一化和静态多帧累积；
- 新增 `roll_lane_bev_refinement.py`，只优化 roll，并 profile-out `b_xy_k`；
- 新增 `calibration_observability.py`，计算曲率、协方差、session 一致性和发布门；
- 将时间偏移和连续时间轨迹加入配置及报告。

建议不要让旧的通用 Canny/LSD 或 TEED 边缘残差直接触发正式外参发布。

## 9. 验证计划

### 9.1 仿真回归

在真实点云和真实图像上独立注入：

```text
LiDAR roll/pitch/yaw: +/-3 deg
Camera roll/pitch/yaw: +/-3 deg
translation: +/-0.10 m
time offset: 多个毫秒级偏移
```

验证：

- 是否能进入正确收敛盆地；
- 公共 LiDAR 修正和相机自身误差是否可分离；
- roll 的估计偏差、方差和触边界率；
- 平移是否被错误吸收到 roll；
- 低纹理和动态场景下是否正确拒绝。

### 9.2 实车验证

至少覆盖：

- 多辆车和多次重新装配；
- 不同温度、振动和冲击后状态；
- 直路、横坡、弯道、匝道和施工区；
- 白天、夜间、雨天和车道线磨损；
- 多个相机视场重叠和低重叠情况。

角度真值应来自独立的高精度测量或受控标定靶，BEV KPI 应来自独立的车道线测量链路，不能用优化目标自身作为真值。

## 10. 论文和量产公开方案依据

- [CalibBEV: LiDAR-Camera Calibration via BEV Alignment, 2026](https://arxiv.org/abs/2608.02309)：相机和 LiDAR 共享 BEV 表示并做粗到细对齐，适合借鉴为 BEV 车道精修结构。
- [Targetless Intrinsics and Extrinsic Calibration with IMU using Continuous-Time Estimation, 2025](https://arxiv.org/abs/2501.02821)：联合估计连续时间轨迹、内外参和时间偏移。
- [MFCalib, 2024](https://arxiv.org/abs/2409.00992)：多类型边缘和 LiDAR 光束模型，适合用于非车道结构辅助因子。
- [YOCO, 2024](https://arxiv.org/abs/2407.18043)：平面共面约束，说明受控平面结构可提供高精度几何约束。
- [Online Extrinsic Camera Calibration for Temporally Consistent IPM, 2020](https://arxiv.org/abs/2008.03722)：使用车道宽度和 IPM 约束 roll，是当前目标最直接的解析参考。
- [Embark US11908164B2](https://patents.google.com/patent/US11908164B2/en)：将道路车道作为虚拟标定靶，联合优化轨迹、地面和 LiDAR-camera 对齐。
- [Lucid WO2024155934A1](https://patents.google.com/patent/WO2024155934A1/en)：工况门控、多 session、一致性检查和事件触发。
- [GM Cruise US20240230866A1](https://patents.google.com/patent/US20240230866A1/en)：在线传感器错位监控、动态/植被过滤和降级策略。
- [Continental/Aumovio US20250139829A1](https://patents.google.com/patent/US20250139829A1/en)：根据可观性和协方差选择性更新参数。

专利反映的是公开的工程思路，不等价于已在量产车型中以相同实现部署；真正量产时仍需使用独立数据和车规测试验证。

## 11. 最终推荐

当前项目下一版的主链路应为：

```text
GNSS + IMU + Odom
    -> 连续时间车体轨迹和重力

LiDAR odometry
    -> 独立 LiDAR 相对运动

Hand-eye
    -> 3 deg 初始误差下的 LiDAR-to-body 粗初始化

相机车道语义 + LiDAR 地面/反射强度 BEV
    -> 虚拟车道靶标

重力先验 + signed BEV lane residual
    -> 一维 roll 精修

多 session + 协方差 + 留出窗口
    -> 0.2 deg 量产发布门
```

在你的约束下，最重要的原则是：**pitch/yaw 由运动和地面结构解决，roll 由 IMU 重力和左右车道 BEV 几何解决，平移只作为窗口内 nuisance 参数，不参与最终发布。**
