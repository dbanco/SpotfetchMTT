# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:11:00 2025

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import KalmanModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import MultiscaleHDoGDetector
from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem

# Generate synthetic data with overlapping pair of spots
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Initial state with starting position and velocity (e.g., from the first measurement)
initial_state = {
    'com': np.zeros(3),  # Initial position from the center of mass at the first time step (only for Spot 1)
    'velocity': np.zeros(3),  # Assuming zero velocity at start
    'acceleration': np.zeros(3),  # Assuming zero acceleration for now
    'bbox': np.zeros(6)
}

sigma_sets = np.array([[1,0.5,0],
                       [1,1,0],
                       [1.5,2,0],
                       [1.5**2,2**2,0],
                       [1.5**3,2**3,0],
                       [1.5**4,2**4,0]])
dsigmas = np.array([0.2, 0.2,0])

# Instantiate Detector, Feature Extractor, Track Model
spot_detector = ThresholdingDetector(threshold=0.1)
# spot_detector = MultiscaleHDoGDetector(sigma_sets=sigma_sets,dsigmas=dsigmas)
track_model= KalmanModel(initial_state, feature_extractor= BasicFeatureExtractor(), process_noise=1e-5, measurement_noise=1e-5, dt=1, )
mht_tracker = MHTTracker(track_model=track_model,n_scan_pruning=4, plot_tree=True)
mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)


# Transition through the data
for scan in range(len(data)):
    mtt_system.process_frame(data[scan],scan)
    
# Show data
scanRange = np.arange(20)
omeRange = np.arange(20)

mht_tracker.tree.plot_all_tracks(data,scanRange,omeRange)

    