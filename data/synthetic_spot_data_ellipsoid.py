# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:11:41 2025

@author: Bahar
"""
import numpy as np
from mtt_framework.feature_extraction import (
    compute_center_of_mass, 
    find_bounding_box, 
    compute_intensity,
    bbox_features,
    compute_velocity
)

# Function to add Gaussian noise
def add_perturbation_to_gaussian(data, noise_strength=0.1):
    noise = np.random.normal(0, noise_strength, data.shape)
    return np.clip(data + noise, 0, 1)  # Keep values between 0 and 1

# 3D Gaussian function with an evolving ellipsoid shape
def gaussian3d(params, theta, eta, omega):
    amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z = params
    exp_term = np.exp(
        -((theta - x_center)**2 / (2 * sx**2) +
          (eta - y_center)**2 / (2 * sy**2) +
          (omega - z_center)**2 / (2 * sz**2))
    )
    return amplitude * exp_term + bg0 + bg1x * theta + bg1y * eta + bg1z * omega

# Process a single spot
def process_spot(current_data, p, prev_com, t, prev_velocity=None):
    CoM = compute_center_of_mass(current_data)
    velocity = compute_velocity(CoM, prev_com) if t > 0 else np.zeros(3)
    bbox = find_bounding_box(current_data > 0)
    bbox_feat = bbox_features(current_data > 0)
    intensity = compute_intensity(current_data)
    
    bbox = bbox if bbox is not None else None
    bbox_feat = bbox_feat if bbox_feat is not None else {'center': None, 'size': None}
    
    state = np.concatenate([CoM, velocity, bbox, np.array(bbox_feat['center']), np.array(bbox_feat['size']), [intensity]])
    return state

# Generate synthetic data (ellipsoid evolving over time)
def generate_synthetic_data_ellipsoid(timesteps=20):
    tta_vals = np.linspace(-10, 10, 40)  
    eta_vals = np.linspace(-10, 10, 40)  
    ome_vals = np.linspace(-10, 10, 20)  
    tth, eta, ome = np.meshgrid(tta_vals, eta_vals, ome_vals, indexing='ij')
    
    data = np.zeros((timesteps, *tth.shape)) 
    
    # Initial spot parameters (starts spherical)
    p1 = [1, -5, -5, -5, 1, 1, 1, 0, 0, 0, 0]  
    
    state_vector = np.zeros((timesteps, 19))  
    prev_com = np.zeros(3)
    prev_velocity = np.zeros(3)
    
    for t in range(timesteps):
        current_data = np.zeros(tth.shape)
        
        # Move the spot over time
        p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]

        # Gradually elongate in the eta direction
        p1[5] = 1 + (t / timesteps) * 4  # sy increases from 1 to 5
        p1[2] = 1 + (t / timesteps) * 1  # sx increases from 1 to 5

        current_data = gaussian3d(p1, tth, eta, ome)  
        state = process_spot(current_data, p1, prev_com, t, prev_velocity)
        state_vector[t] = state  

        data[t] = current_data

    return data, state_vector

