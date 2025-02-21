# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:22:32 2025

@author: Bahar
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import BasicModel
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector

# Select detector "Thresholding" or "HDoG
detector_choice = "Thresholding"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)
    

# Generate synthetic data with overlapping pair of spots
data, state_vector = generate_synthetic_data('single', timesteps=20)

# Initial state with starting position and velocity (e.g., from the first measurement)
initial_state = {
    'position': state_vector[0, 0],  # Initial position from the center of mass at the first time step (only for Spot 1)
    'velocity': np.zeros(3),  # Assuming zero velocity at start
    'acceleration': np.zeros(3)  # Assuming zero acceleration for now
}

# Instantiate the state model
state_model = BasicModel(initial_state, feature_extractor=None)

# Store predicted positions and actual positions for plotting (only for Spot 1)
predicted_positions = []
actual_positions = state_vector[:, 0]  # True positions of the first spot (Spot 1)

# Transition through the data
for t in range(0, len(data)):
    # For each time step, compute the next state assuming constant velocity
    dt = 1  # Assuming a time step of 1
    new_state = state_model.transition(state_model.state, dt)
    
    # Append only the first spot's position (Spot 1)
    predicted_positions.append(new_state['position'])  # Append 3D position for Spot 1

    # extract measurements from the data at the current timestep
    blobs, num_blobs = spot_detector.detect(data[t])
    measurements = state_model.get_measurements(blobs, data[t])

    # Update the state model with the measurements
    for measurement in measurements:
        state_model.update_state(measurement)

# Convert predicted positions and actual positions to numpy arrays for easier handling
predicted_positions = np.array(predicted_positions) 
actual_positions = np.array(actual_positions) 

# Plot Actual vs Predicted Positions in the left panel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left Panel: Plot the actual vs predicted positions (center of mass positions for Spot 1)
ax1.plot(actual_positions[:, 0], actual_positions[:, 1], label="Actual Position Spot 1", color="blue", linestyle='-', marker='o')
ax1.plot(predicted_positions[:, 0], predicted_positions[:, 1], label="Predicted Position Spot 1", color="red", linestyle='--', marker='x')

ax1.set_xlabel('2θ Position')
ax1.set_ylabel('Eta Position')
ax1.set_title('Actual vs Predicted Positions (Spot 1)')
ax1.legend()

# Right Panel: Print the actual vs predicted positions side by side
ax2.axis('off')

# Prepare the text content for the side-by-side display
text = "Time Step | Actual Position Spot 1 | Predicted Position Spot 1\n"
text += "-" * 60 + "\n"
for i in range(len(actual_positions)):  # Ensure we're not going out of bounds
    text += f"{i+1:<10} | {actual_positions[i, 0]:<15.3f} {actual_positions[i, 1]:<15.3f} | {predicted_positions[i, 0]:<15.3f} {predicted_positions[i, 1]:<15.3f}\n"

# Display the text
ax2.text(0, 1, text, fontsize=10, va='top', fontfamily='monospace')

# Show the plot
plt.tight_layout()
plt.show()