# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:08:20 2025

@author: dpqb1
"""
import numpy as np

# Define a custom 3D Gaussian function if hexrd is unavailable
def gaussian3d(params, theta, eta, omega):
    """
    A 3D Gaussian function.
    params: [amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z]
    
    x_center, y_center, z_center coordinate of the center of the Gaussian
    sx, sy, sz are standard deviation (to determine the width)
    bg0,..., bg1z backgrounf terms
    
    amplitude is the peak intensity of the Gaussian
    
    """
    amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z = params
    exp_term = np.exp(
        -((theta - x_center)**2 / (2 * sx**2) +
          (eta - y_center)**2 / (2 * sy**2) +
          (omega - z_center)**2 / (2 * sz**2))
    )
    return amplitude * exp_term + bg0 + bg1x * theta + bg1y * eta + bg1z * omega

def generate_synthetic_data(dataset, timesteps=20):
    """
    Generate synthetic 3D test data with one or two moving spots, with merging occurring at t=10.
   
    Parameters:
    - dataset (string): 
        'single' - Generate data with one moving spot.
        'overlapping pair' - Generate data with two moving spots.
    - timesteps (int): Number of time steps to generate (default 20).
    
    Returns:
    - data (numpy array): 3D synthetic data generated as (t, 2theta, eta, omega)
    - state_vector as a function of time
    """
    # Set up the grid 
    theta_vals = np.linspace(-10, 10, 40)  # range for 2theta
    eta_vals = np.linspace(-10, 10, 40)  # range for eta
    omega_vals = np.linspace(-10, 10, 20)  # range for omega
    
    # Create 3D meshgrid for 2theta, eta, omega
    theta, eta, omega = np.meshgrid(theta_vals, eta_vals, omega_vals, indexing='ij')
    
    # Initialize an array to hold the data for all time steps
    data = np.zeros((timesteps, *theta.shape)) 
    
    # Set initial parameters for the spots (add background parameters)
    p1 = [1, -5, -5, -5, 1, 1, 1, 0, 0, 0, 0]  # Spot 1 parameters (amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z)
    p2 = [1, 5, 5, 5, 1, 1, 1, 0, 0, 0, 0]  # Spot 2 parameters (amplitude, x_center, y_center, z_center, sx, sy, sz, bg0, bg1x, bg1y, bg1z)

    # State vector to store the center positions of spots at each time step
    state_vector = np.zeros((timesteps, 2, 3))  # (t, spot, [x, y, z])
    
    # Define how spots move over time (linear motion)
    for t in range(timesteps):
        if dataset == 'single':
            # Only Spot 1 moves
            p2 = [0] * 11  # other spot should be inactive
            if t < 10:
                # Spot 1 moves away
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]
            else:
                # Spot 1 moves closer towards center
                merge_factor = (t - 10) * 0.1
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
            
        elif dataset == 'overlapping pair':
            # Before merging, spots move apart linearly
            if t < 10:
                p1[1:4] = [-5 + t * 0.5, -5 + t * 0.5, -5 + t * 0.5]  # Spot 1 moves
                p2[1:4] = [5 - t * 0.5, 5 - t * 0.5, 5 - t * 0.5]  # Spot 2 moves
            else:
                # After t=10, spots start merging
                merge_factor = (t - 10) * 0.1  # How much the spots move towards each other
                p1[1:4] = [p1[1] + merge_factor, p1[2] + merge_factor, p1[3] + merge_factor]
                p2[1:4] = [p2[1] - merge_factor, p2[2] - merge_factor, p2[3] - merge_factor]
        
        # Generate the data for this time step
        current_data = np.zeros(theta.shape)
        
        # Pass the 3D coordinates (theta, eta, omega) to the gaussian3d function
        current_data += gaussian3d(p1, theta, eta, omega)  # Pass x, y, z coordinates
        current_data += gaussian3d(p2, theta, eta, omega)  # Pass x, y, z coordinates
        
        # Store the data and the state vector (just positions for now)
        data[t] = current_data
        state_vector[t, 0] = p1[1:4]  # Position of spot 1 at time t
        state_vector[t, 1] = p2[1:4]  # Position of spot 2 at time t
    
    return data, state_vector