# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:50:17 2025

@author: Bahar
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mtt_framework.detection import HDoGDetector_2D
import matplotlib.pyplot as plt
from data.synthetic_spot_data_ellipsoid import generate_synthetic_data_ellipsoid
from mtt_framework.feature_extraction import compute_center_of_mass, find_bounding_box, find_bounding_box_2D
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import center_of_mass
import itertools
import utilities as util

###### VD Simulated HEXD data ######
topPath = r"C:\myDrive\TuftsPhD2022\ProfMillerPhDResearch\ONRProject\VD_sim\VD_sim"
dataFile1 = os.path.join(topPath,r"state_*\simulation\outputs\c103_polycrystal_sample_2_state_*_layer_1_output_data.npz")
dataFile = os.path.join(topPath,r"state_{scan}\simulation\outputs\c103_polycrystal_sample_2_state_{scan}_layer_1_output_data.npz")

params = {}
params['detector'] = 'eiger_sim'
params['imSize'] = (5000,5000)
params['yamlFile'] = os.path.join(topPath,'c103_eiger_calibration.yml') #mruby_0401_eiger_calibration
params['roiSize'] = [30,30,11]


scanRange = np.arange(5)
spotsFiles = []
for state in scanRange:
    spotsDir = os.path.join(topPath,f'state_{state}')
    spotsFiles.append(os.path.join(spotsDir,"spots.npz"))

dataFileSequence = [] 
for scan in scanRange:
    dataFileSequence.append(dataFile.format(scan=scan)) 
    spotData = np.load(spotsFiles[0])

# Spot location
spotInd = 0
x = spotData['Xm'][spotInd]
y = spotData['Ym'][spotInd]
frm = int(spotData['ome_idxs'][spotInd])
eta, tth = util.xyToEtaTthRecenter(x,y,params)

intensity_threshold = 0

# Transition through the data
dims = params['roiSize']
full_data = np.zeros((len(dataFileSequence),dims[0],dims[1],dims[2]))

# Generate range of sigma and dsigma values
sigma_range = np.arange(3, 4)   # Sigma values from 1 to 5
dsigma_range = np.arange(1.1, 1.5)  # Dsigma values from 1 to 5

# Create all possible (sigma, dsigma) pairs
multi_scale_sigmas = list(itertools.product(sigma_range, dsigma_range))

# Convert them into NumPy arrays
multi_scale_sigmas = [(np.array([s, s*2]), np.array([ds, ds])) for s, ds in multi_scale_sigmas]
#print("Generated multi-scale sigma-dsigma pairs:", multi_scale_sigmas)


# Loop over the dataFileSequence
for scan, fname in enumerate(dataFileSequence):
    # Load 3D data (eta, tta, omega)
    data = util.loadPolarROI3D(fname, tth, eta, frm, params)  
    num_frames = data.shape[2]  # Number of omega frames
    
    # Loop through each frame in the 3D data (for each omega rotation angle)
    for ome in range(num_frames):  
        frame = data[:, :, ome]  # Extract 2D slice (eta, tta) for the current omega (rotation angle)
        
        
        # === Create a figure for visualization ===
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))
        
        # === Plot Original Data ===
        ax = axes[0]
        ax.imshow(frame, cmap='viridis')
        ax.set_title(f"Original Data (omega={ome})")
        ax.axis('off')
        
        # === Apply Default Blob Detection ===
        ax = axes[1]
        ax.imshow(frame, cmap='viridis')

        all_blobs = np.zeros_like(frame, dtype=int)

        spot_detector = HDoGDetector_2D(sigmas=np.array([3,3]), dsigmas=np.array([5,5]))
        blobs, num_blobs = spot_detector.detect(frame)

        # Extract unique blobs
        unique_labels = np.unique(blobs)
        unique_labels = unique_labels[unique_labels != 0]

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

                    # Compute CoM
                    x_masked = frame * mask
                    com_tth, com_eta = center_of_mass(x_masked)

                    # Adjust CoM and plot
                    ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

        ax.set_title(f"HDoG (omega={ome})")
        ax.axis('off')

        # === Apply Multi-Scale Blob Detection ===
        ax = axes[2]
        ax.imshow(frame, cmap='viridis')

        all_blobs = np.zeros_like(frame, dtype=int)

        for sigmas, dsigmas in multi_scale_sigmas:
            spot_detector = HDoGDetector_2D(sigmas=sigmas, dsigmas=dsigmas)
            blobs, num_blobs = spot_detector.detect(frame)

            # Merge detected blobs
            all_blobs = np.maximum(all_blobs, blobs) 

        # === Prepare mask visualization ===
        mask_visualization = np.zeros_like(frame, dtype=float)

        # Extract unique blobs after merging
        unique_labels = np.unique(all_blobs)
        unique_labels = unique_labels[unique_labels != 0]

        for blob_label in unique_labels:
            mask = (all_blobs == blob_label)
            mask_visualization[mask] = blob_label  
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

                    # Compute CoM
                    x_masked = frame * mask
                    com_tth, com_eta = center_of_mass(x_masked)

                    # Adjust CoM and plot
                    ax.scatter(com_eta, com_tth, color='red', marker='x', s=100)

        ax.set_title(f"Multi-Scale HDoG (omega={ome})")
        ax.axis('off')

    
        # === Plot the Mask ===
        ax = axes[3]
        ax.imshow(mask_visualization, cmap='gray')  
        ax.set_title(f"Blob Mask (omega={ome})")
        ax.axis('off')

        # Show the plot
        plt.tight_layout()
        plt.show()