# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:39:26 2025

@author: dpqb1
"""

import sys
import os
import json
import numpy as np
import redis
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import KalmanModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem

# Connect to Redis queue
redis_client = redis.Redis(host="redis_queue", port=6379, decode_responses=True)

# Get next available tracking job
job_data = redis_client.lpop("tracking_jobs")
if job_data is None:
    print("No tracking job available. Exiting...")
    sys.exit(0)

job = json.loads(job_data)
start_frame, end_frame = job["start_frame"], job["end_frame"]
print(f"Processing frames {start_frame}-{end_frame}")

# Load synthetic data
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Initialize tracking components
initial_state = {'com': np.zeros(3), 'velocity': np.zeros(3), 'acceleration': np.zeros(3), 'bbox': np.zeros(6)}
spot_detector = ThresholdingDetector(threshold=0.1)
track_model = KalmanModel(initial_state, feature_extractor=BasicFeatureExtractor(), process_noise=1e-5, measurement_noise=1e-5, dt=1)
mht_tracker = MHTTracker(track_model=track_model, n_scan_pruning=2, plot_tree=True)
mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)

# Process assigned frames
for scan in range(start_frame, end_frame + 1):
    mtt_system.process_frame(data[scan], scan)

print(f"Finished processing frames {start_frame}-{end_frame}")
