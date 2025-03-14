# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:05:41 2025

@author: Bahar, Daniel
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
from mtt_framework.detection import HDoGDetector_SKImage, LoGDetector
from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem
import os
import utilities as util

###### VD Simulated HEXD data ######
topPath = r"C:\myDrive\TuftsPhD2022\ProfMillerPhDResearch\ONRProject\VD_sim\VD_sim"
dataFile1 = os.path.join(topPath,r"state_*\simulation\outputs\c103_polycrystal_sample_2_state_*_layer_1_output_data.npz")
dataFile = os.path.join(topPath,r"state_{scan}\simulation\outputs\c103_polycrystal_sample_2_state_{scan}_layer_1_output_data.npz")
# 

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
# For spot 2; Thresholding captures only 4 spots

test_spot= [0, 2]
num_tracks=[2, 4]

spotInd = 3
for spotInd in range(5):
    x = spotData['Xm'][spotInd]
    y = spotData['Ym'][spotInd]
    frm = int(spotData['ome_idxs'][spotInd])
    eta, tth = util.xyToEtaTthRecenter(x,y,params)


    # Initial state with starting position and velocity (e.g., from the first measurement)
    initial_state = {
        'com': np.zeros(3),  # Initial position from the center of mass at the first time step (only for Spot 1)
        'velocity': np.zeros(3),  # Assuming zero velocity at start
        'acceleration': np.zeros(3),  # Assuming zero acceleration for now
        'bbox': np.zeros(6)
    }

    # Instantiate Detector, Feature Extractor, Track Model

    #Original thresholding
    #spot_detector = ThresholdingDetector(threshold=500)
    #Gaussian Filtering + thresholding
    #spot_detector= ThresholdingDetector(threshold=100, use_gaussian_filter=True, sigma=1)
    #Median Filter + thresholding
    #spot_detector = ThresholdingDetector(threshold=20, use_gaussian_filter=False, filter_size=3)

    spot_detector = HDoGDetector_SKImage()
    #spot_detector = HDoGDetector()
    #spot_detector = LoGDetector()

    
    #track_model = BasicModel(initial_state, feature_extractor=BasicFeatureExtractor())
    track_model= KalmanModel(initial_state, feature_extractor= BasicFeatureExtractor(), process_noise=1e-5, measurement_noise=1e-5, dt=1, )
    mht_tracker = MHTTracker(track_model=track_model,n_scan_pruning=3, plot_tree=True)
    mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)


    # Transition through the data
    dims = params['roiSize']
    full_data = np.zeros((len(dataFileSequence),dims[0],dims[1],dims[2]))
    for scan, fname in enumerate(dataFileSequence):
        # Load roi
        data = util.loadPolarROI3D(fname,tth,eta,frm,params)
        # Process with tracker
        mtt_system.process_frame(data,scan)
        # Collect data series
        full_data[scan] = data

    #Plot data
    scanRange = np.arange(len(dataFileSequence))
    omeRange = np.arange(11)
    mht_tracker.tree.plot_all_tracks(full_data,scanRange,omeRange)

    