# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:46:56 2025

@author: dpqb1
"""

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import BasicModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector
from mtt_framework.mht_tracker import MHTTracker

# Select detector "Thresholding" or "HDoG
detector_choice = "Thresholding"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)
    

# Generate synthetic data with overlapping pair of spots
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Initial state with starting position and velocity (e.g., from the first measurement)
initial_state = {
    'com': np.zeros(3),  # Initial position from the center of mass at the first time step (only for Spot 1)
    'velocity': np.zeros(3),  # Assuming zero velocity at start
    'acceleration': np.zeros(3)  # Assuming zero acceleration for now
}

# Instantiate the track model
track_model = BasicModel(initial_state, feature_extractor=BasicFeatureExtractor())

# Initialize tracks and tracker
init_tracks = []
blobs, num_blobs = spot_detector.detect(data[0])
measurements = track_model.get_measurements(blobs, data[0])

time_steps_to_plot = [4,6,8,10,12,14,16]
omeRange = np.arange(20)

mht_tracker = MHTTracker(track_model=track_model)


# Store predicted positions and actual positions for plotting (only for Spot 1)
predicted_positions = []
actual_positions = state_vector[:, 0]  # True positions of the first spot (Spot 1)

# Transition through the data
for t in range(0, len(data)):
    # For each time step, compute the next state assuming constant velocity
    
    if t == 0:
        mht_tracker.initialize_hypothesis_tree(measurements)
        mht_tracker.prediction()
    else:
        # extract measurements from the data at the current timestep
        blobs, num_blobs = spot_detector.detect(data[t])
        print(f'num_blobs = {num_blobs}')
        measurements = track_model.get_measurements(blobs, data[t])
        
        # Do gating and make m2ta matrix
        mht_tracker.current_scan = t
        mht_tracker.gating(measurements)
        print(mht_tracker.m2ta_matrix)
        
        # Generate hypotheses and tree, show tree of option
        mht_tracker.update_hypothesis_tree()
        
        # Evaluate hypotheses
        mht_tracker.evaluate_hypotheses()

        # Prune
        mht_tracker.tree.pruning(3)
        
        # Visualize tree again
        mht_tracker.tree.visualize_hypothesis_tree()
        
        mht_tracker.prediction()
    
# Show data
mht_tracker.tree.plot_all_tracks(data,np.arange(20),omeRange)
    