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
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.feature_extraction import BasicFeatureExtractor
import matplotlib.pyplot as plt
from data.synthetic_spot_data import generate_synthetic_data
import matplotlib.patches as patches
import numpy as np
from mtt_framework.detection_spot import Detection


# Generate test data
data, state_vector = generate_synthetic_data('three spots', timesteps=20)

# Init detector, Select detector "Thresholding" or "HDoG
detector_choice = "Thresholding"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)


#Initial the feature extractor
feature_extractor= BasicFeatureExtractor()

# Define time steps to visualize
time_steps_to_plot = [4, 6, 8, 10, 12, 14, 16]
# Fixed omega
omega_slice = 10  

# Create figure with two rows of subplots
fig, axes = plt.subplots(2, len(time_steps_to_plot), figsize=(15, 6))

for idx, t in enumerate(time_steps_to_plot):
    # Detect blobs in the 3D data
    original_data= np.copy(data[t])
    blobs, num_blobs = spot_detector.detect(data[t])  

    # Extract the omega slice from the detected blobs
    slice_data = blobs[:, :, omega_slice]

    # Initialize feature list
    features_list = []
    unique_labels = np.unique(blobs)
    
    # Exclude background (label 0)
    unique_labels = unique_labels[unique_labels != 0]
    

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        x_masked = original_data * mask
        # Create a Detection object
        detection = Detection(blob_label, mask, x_masked)
        features = feature_extractor.extract_features(detection)
        features_list.append(features)

    # Print extracted features
    print(f"\nTime Step {t}: Detected {num_blobs} Spots")
    for feat in features_list:
        print(f"  -> {feat}")

    #plot the original synthetic spots
    axes[0, idx].imshow(original_data[:, :, omega_slice], 
                         origin='lower', cmap='viridis')
    axes[0, idx].set_title(f'Time t={t} - Raw Data')
    axes[0, idx].set_xlabel('2Theta')
    axes[0, idx].set_ylabel('Eta')

    #plot detected blobs with center of mass
    axes[1, idx].imshow(slice_data, 
                         origin='lower', cmap='viridis')


    for features in features_list:
        com = features["com"]
        bbox = features["bbox"]
        
        if com is not None:
            # Extract coordinates
            com_2theta, com_eta, com_omega = com  
            # Only plot CoM if it belongs to the current ω slice
            if abs(com_omega - omega_slice) < 1:
                
                # Plot center of mass as a red dot
                axes[1, idx].scatter(com_2theta, com_eta, color='red', marker='x', s=50, label="CoM")
                # Annotate spot ID
                #axes[1, idx].text(com_2theta + 5.0, com_eta + 5.0 , f"ID {features['Spot ID']}", 
                                  #color='white', fontsize=8, weight='bold', bbox=dict(facecolor='black', alpha=0.5))

        # Draw bounding box
        if bbox is not None:
            tth_min, tth_max, eta_min, eta_max, ome_min, ome_max = bbox  # Extract only relevant dimensions
            #if ome_min <= omega_slice <= ome_max:
                # Only plot CoM if it belongs to the current ω slice
            
            rect = patches.Rectangle((tth_min, eta_min),  # Adjust extent
                                     (tth_max - tth_min), (eta_max - eta_min),
                                     linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle="--")
            axes[1, idx].add_patch(rect)

    axes[1, idx].set_title(f'Time t={t} - DS')
    axes[1, idx].set_xlabel('2Theta')
    axes[1, idx].set_ylabel('Eta')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()