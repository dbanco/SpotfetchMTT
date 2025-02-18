# -*- coding: utf-8 -*-
"""
spotfetch.MHTMTT

Description:
Module for tracking x-ray diffraction spots in 3D using mulitple hypotheses test
multitarget tracking
Created on Tue Feb 18 09:59:02 2025

@author: dpqb1
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from scipy.ndimage import center_of_mass
from skimage.feature import hessian_matrix
from scipy.ndimage import label

class Measurement:
    """Represents a measurement (candidate spot) in 3D space."""
    
    def __init__(self, x):
        """
        Initialize a measured spot.
        
        Parameters:
        -----------
            - x (array): masked pixel intesity data
        
        Properties:
        - com (array): center of mass (tt,eta,ome) of the blob.
        - bound_box (array): boudning box (tth1,tth2,eta1,eta2,ome1,ome2) of blob 
        - intensity (float): Total intensity of the detected spot.
        """
        self.com = center_of_mass(x)
        self.bound_box = find_bounding_box(x)
        self.intensity = np.sum(x)
        
        pass

class Track:
    """Represents a tracked spot in 3D space."""
    
    def __init__(self, measurement):
        """
        Initialize a detected spot.
        
        Parameters:
        - measurment: A measurement object
        
        Properties:
        - com (array): center of mass (tt,eta,ome) of the blob.
        - bound_box (array): boudning box (tth1,tth2,eta1,eta2,ome1,ome2) of blob 
        - intensity (float): Total intensity of the detected spot.
        - overlap
        - com_velocity
        - Maybe other things related to state transition model?
        
        """
        position = center_of_mass
        pass

    def update(self, measurement):
        """
        Update the track with a new detection.
        
        Parameters:
        - measurment: A measurement object
        """
        
        pass

def find_bounding_box(mask):
  """
  Finds the bounding box coordinates of non-zero pixels in a binary mask.

  Parameters:
  -----------
    mask: A 3D numpy array representing the binary mask.

  Returns:
  --------
    A tuple (tth_min, tth_max, eta_min, eta_max, ome_min, ome_max) representing 
    the bounding box coordinates or None if no non-zero pixels are found.
  """
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  tubs = np.any(mask, axis=2)
  
  if not np.any(rows) or not np.any(cols) or not np.any(tubs):
    return None  # Return None if the mask is empty

  tth_min, tth_max = np.where(rows)[0][[0, -1]]
  eta_min, eta_max = np.where(cols)[0][[0, -1]]
  ome_min, ome_max = np.where(tubs)[0][[0, -1]]
  
  return tth_min, tth_max, eta_min, eta_max, ome_min, ome_max

def detectBlobHDoG(x,params):
    """
    Detects blobs using the Hessian-based Difference of Gaussians (DoG) method.

    Parameters:
    -----------
    x : ndarray
        Input 3D data.

    Returns:
    --------
    tuple
        - blobs : ndarray
            Labeled blob regions.
        - num_blobs : int
            Number of blobs detected.
        - hess_mat : list of ndarray
            Hessian matrices of the DoG result.
    """
    # 1. Compute normalized DoG
    dog_norm = DoG(x, sigma=params['sigmas'], dsigma=params['dsigma'])

    # 2. Pre-segmentation
    hess_mat = hessian_matrix(dog_norm,use_gaussian_derivatives=params['gaussian_derivatives'])
    D1 = np.zeros(hess_mat[0].shape)
    D2 = np.zeros(hess_mat[0].shape)
    D3 = np.zeros(hess_mat[0].shape)
    for i1 in range(hess_mat[0].shape[0]):
        for i2 in range(hess_mat[0].shape[1]):
            for i3 in range(hess_mat[0].shape[2]):
                h_mat = np.array([
                    [hess_mat[0][i1, i2, i3], hess_mat[1][i1, i2, i3], hess_mat[2][i1, i2, i3]],
                    [hess_mat[1][i1, i2, i3], hess_mat[3][i1, i2, i3], hess_mat[4][i1, i2, i3]],
                    [hess_mat[2][i1, i2, i3], hess_mat[4][i1, i2, i3], hess_mat[5][i1, i2, i3]]
                ])
                D1[i1,i2,i3] = h_mat[0,0]
                D2[i1,i2,i3] = np.linalg.det(h_mat[:2,:2])
                D3[i1,i2,i3] = np.linalg.det(h_mat)

    posDefIndicator = (D1 > 0) & (D2 > 0) & (D3 > 0)
    blobs, num_blobs = label(posDefIndicator)
    return blobs, num_blobs, hess_mat

def DoG(x, sigma, dsigma, gamma=2):
    """
    Computes the Difference of Gaussians (DoG) approximation.

    Parameters:
    -----------
    x : ndarray
        Input data.
    sigma : float
        Base Gaussian sigma.
    dsigma : float
        Increment for sigma.
    gamma : float, optional
        Scaling factor, default is 2.

    Returns:
    --------
    ndarray
        Normalized DoG result.
    """
    g1 = gaussian_filter(x, sigma=sigma)
    g2 = gaussian_filter(x, sigma=sigma + dsigma)
    return (g2 - g1) / (np.mean(sigma) * np.mean(dsigma))

def blobsToMeasurements(x,blobs):
    """
    Converts labeled blobs into a list of Measurement objects.

    Parameters:
    -----------
    blobs : ndarray
        Labeled blob regions.

    Returns:
    --------
    measurements:
        List of measurement objects containing candidate spots.
    """
    measurements = []
    unique_labels = np.unique(blobs)
    
    # Exclude background (label 0)
    unique_labels = unique_labels[unique_labels != 0]

    for blob_label in unique_labels:
        mask = (blobs == blob_label)  # Extract binary mask for the current blob
        x_masked = x * mask
        measurement = Measurement(x_masked)
        measurements.append(measurement)

    return measurements

#Edits By Bahar

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


# Select specific time steps for visualization
time_steps_to_plot = [5, 10, 15]

# Plot slices of the data at different time steps
fig, axes = plt.subplots(1, len(time_steps_to_plot), figsize=(15, 5))

data, state_vector = generate_synthetic_data('overlapping pair', timesteps=20)
params= dict()
params['gaussian_derivatives'] = False
params['sigmas'] = np.array([2,2,2])
params['dsigma'] = np.array([1.5,1.5,1.5])

blobs, num_blobs, hess_mat = detectBlobHDoG(data[0],params)



for idx, t in enumerate(time_steps_to_plot):
    # Slice at omega=50 for each time step
    blobs, num_blobs, hess_mat = detectBlobHDoG(data[t],params)
    slice_data = blobs[:,:,10]

    axes[idx].imshow(slice_data, extent=(-10, 10, -10, 10), origin='lower', cmap='viridis')
    axes[idx].set_title(f'Time t={t}')
    axes[idx].set_xlabel('Theta')
    axes[idx].set_ylabel('Eta')

plt.tight_layout()
plt.show()