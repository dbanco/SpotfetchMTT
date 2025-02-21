# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:46:10 2025

3D simulated data test script with Basic Feature Extraction

This will be used to run and verify each individual component of the mtt algorithm
So far we have a tests on:
    - blob detection
    - producing basic measurements, bounding box and CoM
    
Next to do are
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
from mtt_framework.detection import ThresholdingDetector
import matplotlib.pyplot as plt
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.feature_extraction import compute_center_of_mass, find_bounding_box
import matplotlib.patches as patches
import numpy as np

# Generate test data
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)


# Select detector "Thresholding" or "HDoG
detector_choice = "Thresholding"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)
    

# Initialize feature extractor
feature_extractor = BasicFeatureExtractor()

# Define time steps and omega range
time_steps_to_plot = [4, 6, 8, 10, 12, 14, 16]
omeRange = np.arange(4, 14)
Nome = len(omeRange)

# Store axes for future reference
axes_list = []

###Plot Raw Data###
fig, axes = plt.subplots(Nome, len(time_steps_to_plot), figsize=(25, 30))
axes_list.append(axes)

for idx, t in enumerate(time_steps_to_plot):
    for i, ome in enumerate(omeRange):
        # Extract omega slice
        slice_data = data[t, :, :, ome]  
        axes[i, idx].imshow(slice_data, origin='lower', cmap='viridis')
        axes[i, idx].set_title(f'Time t={t}, Ω={ome}')
        axes[i, idx].set_xticks([])
        axes[i, idx].set_yticks([])

plt.tight_layout()
plt.show()


###Plot Detected Blobs###
fig, axes = plt.subplots(Nome, len(time_steps_to_plot), figsize=(25, 30))
axes_list.append(axes)

for idx, t in enumerate(time_steps_to_plot):
    blobs, num_blobs = spot_detector.detect(data[t])
    
    for i, ome in enumerate(omeRange):
        slice_data = blobs[:, :, ome]  # Extract detected blobs slice
        axes[i, idx].imshow(slice_data, origin='lower', cmap='viridis')
        axes[i, idx].set_title(f'Time t={t}, Ω={ome}')
        axes[i, idx].set_xticks([])
        axes[i, idx].set_yticks([])

plt.tight_layout()
plt.show()


###Plot Detected Blobs with CoM & Bounding Boxes###
fig, axes = plt.subplots(Nome, len(time_steps_to_plot), figsize=(25, 30))
axes_list.append(axes)

for idx, t in enumerate(time_steps_to_plot):
    blobs, num_blobs = spot_detector.detect(data[t])

    # Extract features for each detected blob
    features_list = []
    unique_labels = np.unique(blobs)
    
    # Exclude background (label 0)
    unique_labels = unique_labels[unique_labels != 0]

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        bbox = find_bounding_box(mask)
        
        x_masked = data[t] * mask
        com = compute_center_of_mass(x_masked)
        

        # Store detected blob features
        features_list.append({
            "Spot ID": blob_label,
            "Center of Mass": com,
            "Bounding Box": bbox
        })

    for i, ome in enumerate(omeRange):
        slice_data = blobs[:, :, ome]  # Extract detected blobs slice
        axes[i, idx].imshow(slice_data, origin='lower', cmap='viridis')

        # Plot Center of Mass and Bounding Boxes
        for features in features_list:
            com = features["Center of Mass"]
            bbox = features["Bounding Box"]

            if com is not None:
                com_tth, com_eta, com_ome = com  

                # Plot CoM only if it belongs to the current ω slice
                if abs(com_ome - ome) < 1:
                    axes[i, idx].scatter(com_tth, com_eta, color='red', marker='x', s=100)
                    axes[i, idx].text(com_tth + 2.0, com_eta + 2.0, f"ID {features['Spot ID']}",
                                      color='white', fontsize=8, weight='bold', bbox=dict(facecolor='black', alpha=0.5))

            # Draw bounding box
            if bbox is not None:
                if abs(com_ome - ome) < 1:
                    tth_min, tth_max, eta_min, eta_max, ome_min, ome_max = bbox
                    rect = patches.Rectangle((tth_min, eta_min), (tth_max - tth_min), (eta_max - eta_min),
                                         linewidth=2.5, edgecolor='cyan', facecolor='none', linestyle="--")
                    axes[i, idx].add_patch(rect)

        axes[i, idx].set_title(f'Time t={t}, Ω={ome}')
        axes[i, idx].set_xticks([])
        axes[i, idx].set_yticks([])

plt.tight_layout()
plt.show()