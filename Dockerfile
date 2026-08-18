FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scipy opencv-python-headless nuscenes-devkit==1.2.0
WORKDIR /workspace
COPY nuscenes_edge_demo.py joint_body_calibration.py handeye_initializer.py lidar_icp_handeye.py edge_match_inspection.py /workspace/
ENTRYPOINT ["python", "/workspace/lidar_icp_handeye.py"]
