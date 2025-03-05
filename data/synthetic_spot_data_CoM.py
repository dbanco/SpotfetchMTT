# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:08:20 2025

@author: bahar, dpqb1
"""
import numpy as np
from scipy.ndimage import center_of_mass


def gaussian3d(params, tth, eta, ome):
    """
    A 3D Gaussian function.
    params: [amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z]
    
    x_center, y_center, z_center coordinate of the center of the Gaussian
    sx, sy, sz are standard deviation (to determine the width)
    bg0,..., bg1z background terms
    
    amplitude is the peak intensity of the Gaussian
    """
    amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z = params
    exp_term = np.exp(
        -((tth - x_center)**2 / (2 * sx**2) +
          (eta - y_center)**2 / (2 * sy**2) +
          (ome - z_center)**2 / (2 * sz**2))
    )
    return amplitude * exp_term + bg0 + bg1x * tth + bg1y * eta + bg1z * ome

def generate_synthetic_data(dataset, timesteps=20):
    """
    Generate synthetic 3D test data with one or two moving spots, with merging occurring at t=10.
    
    Parameters:
    - dataset (string): 
        'single' - Generate data with one moving spot.
        'overlapping pair' - Generate data with two moving spots.
    - timesteps (int): Number of time steps to generate (default 20).
    
    Returns:
    - data (numpy array): 3D synthetic data generated as (t, tth, eta, ome)
    - state_vector (numpy array): Stores the center of mass of the spots at each time step.
    """
    tta_vals = np.linspace(-10, 10, 40)  # range for tth
    eta_vals = np.linspace(-10, 10, 40)  # range for eta
    ome_vals = np.linspace(-10, 10, 20)  # range for ome
    tth, eta, ome = np.meshgrid(tta_vals, eta_vals, ome_vals, indexing='ij')
    
    data = np.zeros((timesteps, *tth.shape)) 
    p1 = [1, -5, -5, -5, 1, 1, 1, 0, 0, 0, 0]  # Spot 1 parameters
    p2 = [1, 5, 5, 5, 1, 1, 1, 0, 0, 0, 0]  # Spot 2 parameters
    state_vector = np.zeros((timesteps, 2, 3))  # Store CoM for two spots (t, [CoM1_x, CoM1_y, CoM1_z], [CoM2_x, CoM2_y, CoM2_z])
    
    for t in range(timesteps):
        if dataset == 'single':
            p2 = [0] * 11  # Inactive second spot (Spot 2 does not exist)
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]
            else:
                merge_factor = (t - 10) * 0.1
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
            
            # Generate the data for this time step
            current_data = gaussian3d(p1, tth, eta, ome)  # Only Spot 1 contributes
            
            # Calculate center of mass for Spot 1
            CoM1 = center_of_mass(current_data)
            
            # Store the center of mass for Spot 1
            state_vector[t, 0] = CoM1  # Store CoM for Spot 1
            
        elif dataset == 'overlapping pair':
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]
                p2[1:4] = [5 - t * 0.5, 5 - t * 0.5, 5 - t * 0.5]
            else:
                merge_factor = (t - 10) * 0.1
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
                p2[1:4] = [p2[1] - merge_factor, p2[2] - merge_factor, p2[3] - merge_factor]
            
            # Generate the data for this time step
            current_data = np.zeros(tth.shape)
            current_data += gaussian3d(p1, tth, eta, ome)  # Add Spot 1
            current_data += gaussian3d(p2, tth, eta, ome)  # Add Spot 2
            
            # Calculate center of mass for Spot 1
            CoM1 = center_of_mass(gaussian3d(p1, tth, eta, ome))
            
            # Calculate center of mass for Spot 2
            CoM2 = center_of_mass(gaussian3d(p2, tth, eta, ome))
            
            # Store both centers of mass in the state vector (for each spot)
            state_vector[t, 0] = CoM1  # CoM for Spot 1
            state_vector[t, 1] = CoM2  # CoM for Spot 2
        
        data[t] = current_data
    
    return data, state_vector