# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:59:49 2025

@author: B2_LocalUser
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mtt_framework.detection import HDoGDetector_2D
from mtt_framework.detection import ThresholdingDetector
import matplotlib.pyplot as plt
from data.synthetic_spot_data_ellipsoid import generate_synthetic_data_ellipsoid
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.feature_extraction import compute_center_of_mass, find_bounding_box, find_bounding_box_2D
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.ndimage import center_of_mass

# Generate test data
data, state_vector = generate_synthetic_data_ellipsoid(timesteps=20)

intensity_threshold = 0
timestep = 15  # Set the timestep for visualization
omega = 0  # Set the omega frame for visualization

frame = data[timestep][:, :, omega]  # Get the frame for the specific timestep and omega

# === Create a figure with 4 rows and 6 columns ===
fig, axes = plt.subplots(4, 6, figsize=(20, 15))  # 3 rows, 5 columns

# First row: Increasing `sigmas` values, dsigmas is fixed
combined_values = [
    (np.array([1, 1]), np.array([1, 1])),
    (np.array([2, 1]), np.array([1, 1])),
    (np.array([3, 1]), np.array([1, 1])),
    (np.array([4, 1]), np.array([1, 1])),
    (np.array([5, 1]), np.array([1, 1])),
    (np.array([8, 1]), np.array([1, 1]))
]
for i, (sigmas, dsigmas) in enumerate(combined_values):
    ax = axes[0, i]  # Select the axis for the current plot in the first row
    ax.imshow(frame, cmap='viridis')

    # Create the spot detector with varying sigmas
    spot_detector = HDoGDetector_2D(sigmas=sigmas, dsigmas=dsigmas)
    blobs, num_blobs = spot_detector.detect(frame)
    unique_labels = np.unique(blobs)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        bbox = find_bounding_box_2D(mask)

        if bbox is not None:
            tth_min, tth_max, eta_min, eta_max = bbox
            width, height = eta_max - eta_min, tth_max - tth_min
            blob_region = frame[tth_min:tth_max, eta_min:eta_max]

            if np.sum(blob_region) >= intensity_threshold:
                rect = patches.Rectangle(
                    (eta_min, tth_min), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Compute Center of Mass (CoM)
                x_masked = frame * mask
                com_tth, com_eta = center_of_mass(x_masked)

                # Scatter plot CoM at the position of CoM
                ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

    ax.set_title(f"sigmas = {sigmas}, dsigmas = {dsigmas}")
    ax.axis('off')

# Second row: Increasing `dsigmas` values, sigmas is fixed
combined_values = [
    (np.array([1, 1]), np.array([1, 1])),
    (np.array([1, 2]), np.array([1, 1])),
    (np.array([1, 3]), np.array([1, 1])),
    (np.array([1, 4]), np.array([1, 1])),
    (np.array([1, 5]), np.array([1, 1])),
    (np.array([2, 8]), np.array([2, 2]))
]

for i, (sigmas, dsigmas) in enumerate(combined_values):
    ax = axes[1, i]  # Select the axis for the current plot in the second row
    ax.imshow(frame, cmap='viridis')

    # Create the spot detector with varying dsigmas
    spot_detector = HDoGDetector_2D(sigmas=sigmas, dsigmas=dsigmas)
    blobs, num_blobs = spot_detector.detect(frame)
    unique_labels = np.unique(blobs)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        bbox = find_bounding_box_2D(mask)

        if bbox is not None:
            tth_min, tth_max, eta_min, eta_max = bbox
            width, height = eta_max - eta_min, tth_max - tth_min
            blob_region = frame[tth_min:tth_max, eta_min:eta_max]

            if np.sum(blob_region) >= intensity_threshold:
                rect = patches.Rectangle(
                    (eta_min, tth_min), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Compute Center of Mass (CoM)
                x_masked = frame * mask
                com_tth, com_eta = center_of_mass(x_masked)

                # Scatter plot CoM at the position of CoM
                ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

    ax.set_title(f"sigmas = {sigmas}, dsigmas = {dsigmas}")
    ax.axis('off')

# Third row: Increasing both `sigmas` and `dsigmas`
combined_values = [
    (np.array([1, 1]), np.array([1, 1])),
    (np.array([2, 2]), np.array([1, 1])),
    (np.array([3, 3]), np.array([1, 1])),
    (np.array([4, 4]), np.array([1, 1])),
    (np.array([5, 5]), np.array([1, 1])),
    (np.array([8, 8]), np.array([1, 1]))
]

for i, (sigmas, dsigmas) in enumerate(combined_values):
    ax = axes[2, i]  # Select the axis for the current plot in the third row
    ax.imshow(frame, cmap='viridis')

    # Create the spot detector with varying sigmas and dsigmas
    spot_detector = HDoGDetector_2D(sigmas=sigmas, dsigmas=dsigmas)
    blobs, num_blobs = spot_detector.detect(frame)
    unique_labels = np.unique(blobs)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        bbox = find_bounding_box_2D(mask)

        if bbox is not None:
            tth_min, tth_max, eta_min, eta_max = bbox
            width, height = eta_max - eta_min, tth_max - tth_min
            blob_region = frame[tth_min:tth_max, eta_min:eta_max]

            if np.sum(blob_region) >= intensity_threshold:
                rect = patches.Rectangle(
                    (eta_min, tth_min), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Compute Center of Mass (CoM)
                x_masked = frame * mask
                com_tth, com_eta = center_of_mass(x_masked)

                # Scatter plot CoM at the position of CoM
                ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

    ax.set_title(f"sigmas = {sigmas}, dsigmas = {dsigmas}")
    ax.axis('off')

# Fourth row: Increasing both `sigmas` and `dsigmas`
combined_values = [
    (np.array([1, 1]), np.array([1, 1])),
    (np.array([2, 2]), np.array([2, 2])),
    (np.array([3, 3]), np.array([2, 2])),
    (np.array([4, 4]), np.array([2, 2])),
    (np.array([5, 5]), np.array([2, 2])),
    (np.array([8, 8]), np.array([2, 2]))
]

for i, (sigmas, dsigmas) in enumerate(combined_values):
    ax = axes[3, i]  # Select the axis for the current plot in the third row
    ax.imshow(frame, cmap='viridis')

    # Create the spot detector with varying sigmas and dsigmas
    spot_detector = HDoGDetector_2D(sigmas=sigmas, dsigmas=dsigmas)
    blobs, num_blobs = spot_detector.detect(frame)
    unique_labels = np.unique(blobs)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for blob_label in unique_labels:
        mask = (blobs == blob_label)
        bbox = find_bounding_box_2D(mask)

        if bbox is not None:
            tth_min, tth_max, eta_min, eta_max = bbox
            width, height = eta_max - eta_min, tth_max - tth_min
            blob_region = frame[tth_min:tth_max, eta_min:eta_max]

            if np.sum(blob_region) >= intensity_threshold:
                rect = patches.Rectangle(
                    (eta_min, tth_min), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Compute Center of Mass (CoM)
                x_masked = frame * mask
                com_tth, com_eta = center_of_mass(x_masked)

                # Scatter plot CoM at the position of CoM
                ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

    ax.set_title(f"sigmas = {sigmas}, dsigmas = {dsigmas}")
    ax.axis('off')
# Show the plot for this frame
plt.tight_layout()
plt.show()