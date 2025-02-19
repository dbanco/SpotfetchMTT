# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:56:13 2025

3D simulated data test script

This will be used to run and verify each individual component of the mtt algorithm
So far we have a tests on:
    - blob detection
Next to do are
    - producing measurements
    - initializing tracks
    - gating, producing meaure-to-track-associations (M2TAs)
    - hypothesis generation from M2TAs
    - hypothesis evaluation
    - track updating
    - 

@author: Bahar, Daniel
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mtt_framework.detection import HDoGDetector
import matplotlib.pyplot as plt
from data.synthetic_spot_data import generate_synthetic_data

# Generate test data
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Init detector
spot_detector = HDoGDetector()

# Plot slices of the data at different time steps
time_steps_to_plot = [5, 10, 15]
fig, axes = plt.subplots(1, len(time_steps_to_plot), figsize=(15, 5))
for idx, t in enumerate(time_steps_to_plot):
    blobs, num_blobs = spot_detector.detect(data[t])
    slice_data = blobs[:,:,10]

    axes[idx].imshow(slice_data, extent=(-10, 10, -10, 10), origin='lower', cmap='viridis')
    axes[idx].set_title(f'Time t={t}')
    axes[idx].set_xlabel('Theta')
    axes[idx].set_ylabel('Eta')

plt.tight_layout()
plt.show()