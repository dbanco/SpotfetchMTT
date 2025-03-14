# -*- coding: utf-8 -*-
"""
Created on Thur Mar 13 22:50:25 2025

@author: Bahar, Daniel

"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import utilities as util
from mtt_framework.state_model import BasicModel
from mtt_framework.state_model import KalmanModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import (
    ThresholdingDetector,
    HDoGDetector,
    HDoGDetector_SKImage
    )

from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem


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
# For spot 2; Thresholding captures only 4 spots
#test_spot= [0, 1, 2, 3, 4 , 5]
#num_tracks=[2, 4]

# Instantiate Detector
detectors = {
    "HDoGDetector_SKImage": HDoGDetector_SKImage(),
    "HDoGDetector": HDoGDetector(),
    "Thresholding_500": ThresholdingDetector(threshold=500),
    "Thresholding + Median_500": ThresholdingDetector(threshold=500, use_gaussian_filter=False, filter_size=3),
    "Thresholding_200": ThresholdingDetector(threshold=200),
    "Thresholding + Median_200": ThresholdingDetector(threshold=200, use_gaussian_filter=False, filter_size=3),
    "Thresholding_100": ThresholdingDetector(threshold=100),
    "Thresholding + Median_100": ThresholdingDetector(threshold=100, use_gaussian_filter=False, filter_size=3)
} 

# Tracking results
results = {}
# Process each detector
for detector_name, spot_detector in detectors.items():
    results[detector_name] = {}

    for spotInd in range(5):  
        x, y = spotData['Xm'][spotInd], spotData['Ym'][spotInd]
        frm = int(spotData['ome_idxs'][spotInd])
        eta, tth = util.xyToEtaTthRecenter(x, y, params)

        # Initial State
        initial_state = {
            'com': np.zeros(3),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'bbox': np.zeros(6)
        }

        # Instantiate Tracker Model
        track_model = KalmanModel(
            initial_state, 
            feature_extractor=BasicFeatureExtractor(), 
            process_noise=1e-5, 
            measurement_noise=1e-5, 
            dt=1
        )
        mht_tracker = MHTTracker(track_model=track_model, n_scan_pruning=3, plot_tree=True)
        mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)

        # Process Data
        dims = params['roiSize']
        full_data = np.zeros((len(dataFileSequence), dims[0], dims[1], dims[2]))

        for scan, fname in enumerate(dataFileSequence):
            data = util.loadPolarROI3D(fname, tth, eta, frm, params)
            mtt_system.process_frame(data, scan)
            full_data[scan] = data

        #Plots
        scanRange = np.arange(len(dataFileSequence))
        omeRange = np.arange(11)
        #example vlim usage = vlim= (0,20)
        mht_tracker.tree.plot_all_tracks(full_data,scanRange,omeRange, vlim= None)
        # Store result
        results[detector_name][spotInd] = mht_tracker.tree.next_track_id

# Print Table
df = pd.DataFrame(results)
print(df.T)
    