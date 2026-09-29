FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scipy opencv-python-headless PyYAML nuscenes-devkit==1.2.0 \
    && pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /workspace
RUN mkdir -p /workspace/models && python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/xavysp/TEED/40fa4b1391dc6424f88989d0ca75d5b592c8681d/checkpoints/BIPED/5/5_model.pth', '/workspace/models/teed_biped_epoch5.pth')" \
    && echo '0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba  /workspace/models/teed_biped_epoch5.pth' | sha256sum -c -
COPY nuscenes_edge_demo.py joint_body_calibration.py handeye_initializer.py lidar_icp_handeye.py edge_match_inspection.py stacked_lidar_demo.py teed_model.py teed_ground_inspection.py teed_edge_cache.py teed_stacked_refinement.py teed_vertical_roll_refinement.py vertical_edge_inspection.py vertical_instance_tracker.py vertical_track_projection.py vertical_track_teed_consistency.py tracked_vertical_roll_refinement.py mfcalib_python.py mfcalib_config.example.yaml /workspace/
ENTRYPOINT ["python", "/workspace/lidar_icp_handeye.py"]
