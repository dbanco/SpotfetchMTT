# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:28:33 2025

Test of detection methods on 3D data containing different spot shapes

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector
import matplotlib.pyplot as plt
from data.synthetic_spot_data import generate_synthetic_data


# Generate test data
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Init detectors
detector_list = [ThresholdingDetector(threshold=0.1),HDoGDetector()]
axes_list = []
time_steps_to_plot = [4,6,8,10,12,14,16]
omeRange = np.arange(4,14)
Nome = len(omeRange)
    
# Plot slices of the data at different time steps
for spot_detector in detector_list:
    fig, axes = plt.subplots(Nome, len(time_steps_to_plot), figsize=(45, 35))
    axes_list.append(axes)
    for idx, t in enumerate(time_steps_to_plot):
        blobs, num_blobs = spot_detector.detect(data[t])
        for i, ome in enumerate(omeRange):
            slice_data = blobs[:,:,ome]
            axes_list[-1][i][idx].imshow(slice_data,origin='lower', cmap='viridis')

    plt.tight_layout()
    plt.show()
    
# Show raw data
fig, axes = plt.subplots(Nome, len(time_steps_to_plot), figsize=(45, 35))
axes_list.append(axes)
for idx, t in enumerate(time_steps_to_plot):
    for i, ome in enumerate(omeRange):
        slice_data = data[t,:,:,ome]
        axes_list[-1][i][idx].imshow(slice_data,origin='lower', cmap='viridis')


plt.tight_layout()
plt.show()
