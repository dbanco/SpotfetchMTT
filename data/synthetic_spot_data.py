# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:08:20 2025
Updated on Mon Mar  3 22:07:01 2025, add three spots, state vectors

@author: Bahar, Daniel
"""

import numpy as np
from mtt_framework.feature_extraction import (
    compute_center_of_mass, 
    find_bounding_box, 
    compute_intensity,
    bbox_features,
    compute_velocity
)

def gaussian3d(params, theta, eta, omega):
    """
    A 3D Gaussian function.
    """
    amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z = params
    exp_term = np.exp(
        -((theta - x_center)**2 / (2 * sx**2) +
          (eta - y_center)**2 / (2 * sy**2) +
          (omega - z_center)**2 / (2 * sz**2))
    )
    return amplitude * exp_term + bg0 + bg1x * theta + bg1y * eta + bg1z * omega

def process_spot(current_data, p, prev_com, t, prev_velocity=None):
    """
    Process a single spot and calculate its features.
    """
    CoM = compute_center_of_mass(current_data)
    velocity = compute_velocity(CoM, prev_com) if t > 0 else np.zeros(3)
    bbox = find_bounding_box(current_data > 0)
    bbox_feat = bbox_features(current_data > 0)
    intensity = compute_intensity(current_data)
    
    bbox = bbox if bbox is not None else None
    bbox_feat = bbox_feat if bbox_feat is not None else {'center': None, 'size': None}
    
    state = np.concatenate([CoM, velocity, bbox, np.array(bbox_feat['center']), np.array(bbox_feat['size']), [intensity]])
    return state

def generate_synthetic_data(dataset, timesteps=20):
    """
    Generate synthetic 3D test data with moving spots, with merging at t=10.
    """
    tta_vals = np.linspace(-10, 10, 40)  
    eta_vals = np.linspace(-10, 10, 40)  
    ome_vals = np.linspace(-10, 10, 20)  
    tth, eta, ome = np.meshgrid(tta_vals, eta_vals, ome_vals, indexing='ij')
    
    data = np.zeros((timesteps, *tth.shape)) 
    p1 = [1, -5, -5, -5, 1, 1, 1, 0, 0, 0, 0]  
    p2 = [1, 5, 5, 5, 1, 1, 1, 0, 0, 0, 0]  
    p3 = [1, 0, 8, -8, 1, 1, 1, 0, 0, 0, 0]  

    num_spots = {'single': 1, 'overlapping pair': 2, 'three spots': 3}[dataset]
    state_vector = np.zeros((timesteps, num_spots, 19))  # 19 features per spot (CoM, velocity, bbox, bbox center, size, intensity)
    
    prev_coms = [np.zeros(3)] * num_spots
    prev_velocities = [np.zeros(3)] * num_spots
    
    for t in range(timesteps):
        current_data = np.zeros(tth.shape)
        
        if dataset == 'single':
            p2 = [0] * 11  # Inactive second spot
            p3 = [0] * 11  # Inactive third spot
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]
            else:
                merge_factor = (t - 10) * 0.1
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
                
            current_data = gaussian3d(p1, tth, eta, ome)  
            # Process the first spot
            state1 = process_spot(current_data, p1, prev_coms[0], t, prev_velocities[0])
            state_vector[t, 0] = state1  

        elif dataset == 'overlapping pair':
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]  # Spot 1 moves
                p2[1:4] = [5 - t * 0.5, 5 - t * 0.5, 5 - t * 0.5]  # Spot 2 moves
            else:
                merge_factor = (t - 10) * 0.1  # How much the spots move towards each other
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
                p2[1:4] = [p2[1] - merge_factor, p2[2] - merge_factor, p2[3] - merge_factor]
            
            current_data = gaussian3d(p1, tth, eta, ome)  
            # Process the first spot
            state1 = process_spot(current_data, p1, prev_coms[0], t, prev_velocities[0])
            state_vector[t, 0] = state1  
            
            current_data += gaussian3d(p2, tth, eta, ome)  
            # Process the second spot
            state2 = process_spot(current_data, p2, prev_coms[1], t, prev_velocities[1])
            state_vector[t, 1] = state2  

        elif dataset == 'three spots':
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]
                p2[1:4] = [5 - t * 0.5, 5 - t * 0.5, 5 - t * 0.5]
                p3[1:4] = [1, 6 - t * 0.3, -6 + t * 0.3]  # Spot 3 moves but stays slightly offset
            else:
                merge_factor = (t - 10) * 0.1
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
                p2[1:4] = [p2[1] - merge_factor, p2[2] - merge_factor, p2[3] - merge_factor]
                p3[1:4] = [1, max(2, 6 - t * 0.1), max(-2, -6 + t * 0.1)]  # p3 slows down, keeping distance
            
            current_data = gaussian3d(p1, tth, eta, ome)
            # Process the first spot
            state1 = process_spot(current_data, p1, prev_coms[0], t, prev_velocities[0])
            state_vector[t, 0] = state1  
             
            
            current_data += gaussian3d(p2, tth, eta, ome)  
            # Process the second spot
            state2 = process_spot(current_data, p2, prev_coms[1], t, prev_velocities[1])
            state_vector[t, 1] = state2  

            current_data += gaussian3d(p3, tth, eta, ome)  
            # Process the third spot
            state3 = process_spot(current_data, p3, prev_coms[2], t, prev_velocities[2])
            state_vector[t, 2] = state3  

        # Store the current data
        data[t] = current_data

    return data, state_vector