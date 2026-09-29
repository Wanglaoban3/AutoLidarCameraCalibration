# auto-extrinsics-adjusting

基于 nuScenes mini 的工作区，两大主题：**BEV 地面投影 / 跨视角匹配** 与
**相机-LiDAR pitch/roll 外参恢复**。环境：`H:\miniconda3\envs\yt`；
数据：`H:\datasets\nuscenes-mini`（已解压）、`H:\datasets\can_bus_extract`。

## 一、BEV 投影与匹配脚本

| 脚本 | 说明 |
|---|---|
| `bev_stitch.py` | 6 相机地面反投影 BEV 拼接主脚本（cam_out 示例 `outputs\bev\bev_stitch_*.png`） |
| `bev_random_vis.py` | 随机 20 帧 "BEV + 6 原始视角" 组合图（`outputs\bev\frames\frame_XX.png`） |
| `bev_prevnext.py` | 相邻帧/时移帧管线（含 `walk_chain`、`segment_ground`、`quat_to_mat` 等被其他脚本复用的工具函数） |
| `bev_lane_match.py` | 车道线（顶帽+颜色）/路缘提取 → BEV 投影 → 线段关联（匈牙利） |
| `bev_match_aliked.py` | ALIKED+LightGlue 跨视角匹配 → 已知位姿极线过滤 → BEV 投影 |
| `bev_epipolar_vis.py` | 极线优化残差柱状图 + 原图匹配点可视化 |
| `bev_aliked_image_vis.py` | 匹配点画回原图（绿=内点 红=拒绝） |
| `bev_opt_epipolar.py` | 极线一致性外参精炼（前视-前左，12 帧，3.68px→0.44px） |

要点：nuScenes ego 系 x=前 y=左；跨帧投影必须逐帧用 ego_pose 姿态补偿
（`g = R_le(帧)ᵀ R_geᵀ ẑ` 全局链）；地面 BEV 对齐残差主要来自地面平面
假设与 ego 姿态变化，不是外参。

## 二、外参恢复（`extrinsic_recovery\`，详见该目录 README.md）

以官方标定为 GT，pitch/roll 注入 ±2° 噪声后恢复：

- 相机：建筑竖线（YOLO 建筑分割 + TEED 原生分辨率边缘 + 梯度亚像素定位）
  + 方向空间 RANSAC/IRLS + 逐帧 IMU 重力 + 逆方差多帧聚合 →
  1.666° → **0.507°**
- LiDAR：地面多格法向锚点 + 立面正交弱约束 + 切平面 2 参数优化 →
  1.486° → **0.364°**

## 三、输出目录

- `outputs\bev\`：拼接 BEV、单相机 BEV
- `outputs\bev\frames\`：20 帧组合可视化、邻帧/地面/边缘各系列
- `outputs\bev\frames\frame18_lane_match\`：车道线/路缘 BEV、地平线
  匹配、2D 阶段图、真彩 BEV
- `extrinsic_recovery\results\`：恢复外参 npy、竖线/平面可视化

## 注意事项

- `bev_prevnext.py` 与 `bev_lane_match.py` 含被复用的工具函数，勿单独删除。
- LiDAR sweep 若不在解压目录，相关脚本会自动从
  `H:\datasets\v1.0-mini\v1.0-mini.tar` 按需解压。
- LightGlue 源码在 `H:\projects\LightGlue-main`（sys.path 引入），
  ALIKED/LightGlue 权重已缓存在 torch hub 目录。
