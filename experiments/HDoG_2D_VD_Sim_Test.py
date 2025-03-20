# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:31:41 2025

@author: B2_LocalUser
"""

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import BasicModel
from mtt_framework.state_model import KalmanModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector
from mtt_framework.detection import HDoGDetector_SKImage
from mtt_framework.detection import HDoGDetector_2D
from mtt_framework.detection import HDoGDetector_SKImmage_2D
from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem
from mtt_framework.feature_extraction import find_bounding_box_2D
import os
import utilities as util
from skimage.feature import blob_log, blob_dog, blob_doh, canny
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import disk
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import center_of_mass

###### VD Simulated HEXD data ######
topPath = r"C:\myDrive\TuftsPhD2022\ProfMillerPhDResearch\ONRProject\VD_sim\VD_sim"
dataFile1 = os.path.join(topPath,r"state_*\simulation\outputs\c103_polycrystal_sample_2_state_*_layer_1_output_data.npz")
dataFile = os.path.join(topPath,r"state_{scan}\simulation\outputs\c103_polycrystal_sample_2_state_{scan}_layer_1_output_data.npz")

# Select detector "Thresholding" or "HDoG
detector_choice = "HDoG_2D"
if detector_choice == "HDoG":
    spot_detector = HDoGDetector()
elif detector_choice == "Thresholding":
    spot_detector = ThresholdingDetector(threshold=0.1)
elif detector_choice == "HDoG_2D":
    spot_detector = HDoGDetector_2D()

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


# Transition through the data
dims = params['roiSize']
full_data = np.zeros((len(dataFileSequence),dims[0],dims[1],dims[2]))

intensity_threshold= 0
# Loop through the dataFileSequence
for scan, fname in enumerate(dataFileSequence):
    # Load 3D data (eta, tta, omega)
    data = util.loadPolarROI3D(fname, tth, eta, frm, params)  
    num_frames = data.shape[2]  # Number of omega frames
    
    # Loop through each frame (omega)
    for ome in range(num_frames):  
        # Extract 2D slice for the current omega (eta, tta)
        frame = data[:, :, ome]  # Current omega slice
        frame_gray = rgb2gray(frame)  # Convert to grayscale if needed
        
        # === Perform blob detection ===
        # 1. Laplacian of Gaussian (LoG) method
        blobs_log = blob_log(frame_gray, max_sigma=2, num_sigma=1, threshold=0.1)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)  # Adjust radius
        
        # 2. Difference of Gaussian (DoG) method
        blobs_dog = blob_dog(frame_gray, max_sigma=2, threshold=0.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)  # Adjust radius
        
        # 3. Hessian-based Difference of Gaussian (HDoG)
        spot_detector_1 = HDoGDetector_2D(sigmas=np.array([3,3]), dsigmas=np.array([5,5]))
        blob_hdog, num_blobs = spot_detector_1.detect(frame_gray)
        
        
        # 4. Hessian-based Difference of Gaussian (HDoG) SKImage
        #Best so far is sigma with 3 and dsigma with 5
        spot_detector_2 = HDoGDetector_SKImmage_2D(sigmas=np.array([3,3]), dsigmas=np.array([5,5]))
        blob_hdog_ski, num_blobs_ski = spot_detector_2.detect(frame_gray)
        
        
        # 4. Hessian-based Difference of Gaussian (HDoG) SKImage
        #Best so far is sigma with 3 and dsigma with 5
        spot_detector_3 = ThresholdingDetector(threshold=0.1)
        blob_thresh, num_blobs_thresh = spot_detector_3.detect(frame_gray)
        
        
        # === Create a new figure for each omega ===
        fig, axes = plt.subplots(1, 6, figsize=(20, 5))  # 1 row, 4 columns for each method
        
        # === Plot the Original Data ===
        ax = axes[0]  # First subplot for original data
        ax.imshow(frame_gray, cmap='viridis')
        ax.set_title(f"Original Data (omega={ome})")
        ax.axis('off')  # Hide axis

        # === Plot the LoG blobs (Yellow circles) ===
        ax = axes[1]  # Second subplot for LoG
        ax.imshow(frame_gray, cmap='viridis')
        for blob in blobs_log:
            y, x, r = blob
            y_min, y_max = int(y - r), int(y + r)
            x_min, x_max = int(x - r), int(x + r)
            blob_region = frame[y_min:y_max, x_min:x_max]
            if np.sum(blob_region) >= intensity_threshold:
                ax.add_patch(Circle((x, y), r, color='yellow', linewidth=2, fill=False))
        ax.set_title(f"Blob LoG (omega={ome})")
        ax.axis('off')

        # === Plot the DoG blobs (Lime circles) ===
        ax = axes[2]  # Third subplot for DoG
        ax.imshow(frame_gray, cmap='viridis')
        for blob in blobs_dog:
            y, x, r = blob
            y_min, y_max = int(y - r), int(y + r)
            x_min, x_max = int(x - r), int(x + r)
            blob_region = frame[y_min:y_max, x_min:x_max]
            if np.sum(blob_region) >= intensity_threshold:
                ax.add_patch(Circle((x, y), r, color='lime', linewidth=2, fill=False))
        ax.set_title(f"Blob DoG (omega={ome})")
        ax.axis('off')

        # === Plot the HDoG blobs with bounding boxes and center of mass ===
        ax = axes[3]  # Fourth subplot for HDoG with bounding boxes
        ax.imshow(frame_gray, cmap='viridis')

        unique_labels = np.unique(blob_hdog)
        unique_labels = unique_labels[unique_labels != 0]  # Ignore background (label=0)

        for blob_label in unique_labels:
            mask = (blob_hdog == blob_label)  # Create binary mask for the blob
            bbox = find_bounding_box_2D(mask)  # Get bounding box (tth_min, tth_max, eta_min, eta_max)

            if bbox is not None:
                # Extract bounding box coordinates
                tth_min, tth_max, eta_min, eta_max = bbox

                # Convert bounding box format for Matplotlib
                x_min, y_min = eta_min, tth_min
                x_max, y_max = eta_max, tth_max
                width, height = eta_max - eta_min, tth_max - tth_min
                blob_region = frame[y_min:y_max, x_min:x_max]

                # Plot bounding box (Red Rectangle)
                if np.sum(blob_region) >= intensity_threshold:
                    ax.add_patch(Rectangle(
                        (x_min, y_min), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    ))

                    # Compute and plot center of mass (Red Dot)
                    x_masked = frame * mask  # Apply mask to the image
                    com = center_of_mass(x_masked)
                    ax.plot(com[1], com[0], 'ro', markersize=10)  # Plot center of mass as a red dot
                    
        ax.set_title(f"HDoG (omega={ome})")
        ax.axis('off')  # Hide axis       
        
        
        # === Plot the HDoG blobs SKImage with bounding boxes and center of mass ===
        ax = axes[4]  # Fourth subplot for HDoG with bounding boxes
        ax.imshow(frame_gray, cmap='viridis')

        unique_labels_2 = np.unique(blob_hdog_ski)
        unique_labels_2 = unique_labels_2[unique_labels_2 != 0]  # Ignore background (label=0)

        for blob_label in unique_labels_2:
            mask = (blob_hdog_ski == blob_label)  # Create binary mask for the blob
            bbox = find_bounding_box_2D(mask)  # Get bounding box (tth_min, tth_max, eta_min, eta_max)

            if bbox is not None:
                # Extract bounding box coordinates
                tth_min, tth_max, eta_min, eta_max = bbox

                # Convert bounding box format for Matplotlib
                x_min, y_min = eta_min, tth_min
                x_max, y_max = eta_max, tth_max
                width, height = eta_max - eta_min, tth_max - tth_min
                blob_region = frame[y_min:y_max, x_min:x_max]

                # Plot bounding box (Red Rectangle)
                if np.sum(blob_region) >= intensity_threshold:
                    ax.add_patch(Rectangle(
                        (x_min, y_min), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    ))

                    # Compute and plot center of mass (Red Dot)
                    x_masked = frame * mask  # Apply mask to the image
                    com = center_of_mass(x_masked)
                    ax.plot(com[1], com[0], 'ro', markersize=10)  # Plot center of mass as a red dot

        ax.set_title(f"HDoG SKImage (omega={ome})")
        ax.axis('off')  # Hide axis
        
        
        # === Plot the HDoG blobs with bounding boxes and center of mass ===
        ax = axes[5]  # Fourth subplot for HDoG with bounding boxes
        ax.imshow(frame_gray, cmap='viridis')

        unique_labels_3 = np.unique(blob_thresh)
        unique_labels_3 = unique_labels_3[unique_labels_3 != 0]  # Ignore background (label=0)

        for blob_label in unique_labels_3:
            mask = (blob_thresh == blob_label)  # Create binary mask for the blob
            bbox = find_bounding_box_2D(mask)  # Get bounding box (tth_min, tth_max, eta_min, eta_max)

            if bbox is not None:
                # Extract bounding box coordinates
                tth_min, tth_max, eta_min, eta_max = bbox

                # Convert bounding box format for Matplotlib
                x_min, y_min = eta_min, tth_min
                x_max, y_max = eta_max, tth_max
                width, height = eta_max - eta_min, tth_max - tth_min
                blob_region = frame[y_min:y_max, x_min:x_max]

                # Plot bounding box (Red Rectangle)
                if np.sum(blob_region) >= intensity_threshold:
                    ax.add_patch(Rectangle(
                        (x_min, y_min), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    ))

                    # Compute and plot center of mass (Red Dot)
                    x_masked = frame * mask  # Apply mask to the image
                    com = center_of_mass(x_masked)
                    ax.plot(com[1], com[0], 'ro', markersize=10)  # Plot center of mass as a red dot

        ax.set_title(f"Threshold (omega={ome})")
        ax.axis('off')  # Hide axis

        # Adjust layout for better spacing between subplots
        plt.tight_layout()

        # Show the figure for the current omega
        plt.show()