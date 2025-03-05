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
from data.synthetic_spot_data_CoM import generate_synthetic_data
from mtt_framework.state_model import KalmanModel
from mtt_framework.state_model import BasicModel
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector
from mtt_framework.feature_extraction import BasicFeatureExtractor

# Generate synthetic data for overlapping spots
data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)

# Initialize state using first spot
dt = 1
initial_state = {
    'position': state_vector[0, 0],  
    'velocity': (state_vector[1, 0] - state_vector[0, 0]) / dt,  # Estimate initial velocity
    'acceleration': np.zeros(3)
}

# Select detector ("Thresholding" or "HDoG")
detector_choice = "Thresholding"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)
    

feature_extractor= BasicFeatureExtractor()
state_model_choice= 'Basic'
if state_model_choice == 'Basic':
    state_model = BasicModel(initial_state, feature_extractor=feature_extractor)
elif state_model_choice == 'KalmanFilter':
    state_model= KalmanModel(initial_state, feature_extractor=feature_extractor, process_noise=1e-5, measurement_noise=1e-5, dt=1)


# Store predicted and actual positions
predicted_positions = []
actual_positions = state_vector[:, 0]  # True positions of Spot 1

# Process each timestep
for t in range(len(data)):
    # Predict next state
    new_state = state_model.transition(state_model.state, dt)

    # Store predicted position
    predicted_positions.append(new_state['position'])

    # Detect spots at the current timestep
    blobs, num_blobs = spot_detector.detect(data[t])
    measurements = state_model.get_measurements(blobs, data[t])

    # Select the closest measurement to the predicted position
    if measurements:
        predicted_pos = new_state['position']
        distances = [np.linalg.norm(predicted_pos - m['com']) for m in measurements]
        best_match = np.argmin(distances)  # Index of best measurement

        # Update the state using the best-matched measurement
        state_model.update_state(measurements[best_match], dt=1)

# Convert predictions to numpy array
predicted_positions = np.array(predicted_positions)

# Plot Actual vs Predicted Positions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left Panel: Actual vs Predicted positions
ax1.plot(actual_positions[:, 0], actual_positions[:, 1], label="Actual Spot 1", color="blue", marker='o')
ax1.plot(predicted_positions[:, 0], predicted_positions[:, 1], label="Predicted Spot 1", color="red", linestyle='--', marker='x')
ax1.set_xlabel('2θ Position')
ax1.set_ylabel('Eta Position')
ax1.set_title('Tracking Spot 1 in Overlapping Case')
ax1.legend()

# Right Panel: Print actual vs predicted values
ax2.axis('off')
text = "Time Step | Actual Position (Spot 1) | Predicted Position (Spot 1)\n"
text += "-" * 60 + "\n"
for i in range(len(actual_positions)):
    text += f"{i+1:<10} | {actual_positions[i, 0]:<15.3f} {actual_positions[i, 1]:<15.3f} | {predicted_positions[i, 0]:<15.3f} {predicted_positions[i, 1]:<15.3f}\n"

ax2.text(0, 1, text, fontsize=10, va='top', fontfamily='monospace')

plt.tight_layout()
plt.show()