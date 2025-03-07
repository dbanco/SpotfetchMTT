# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 20:20:20 2025

@author: dpqb1
"""
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import utilities as util
from data.synthetic_spot_data import generate_synthetic_data
from mtt_framework.state_model import BasicModel
from mtt_framework.state_model import KalmanModel
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.detection import HDoGDetector
from mtt_framework.mht_tracker import MHTTracker
from mtt_system import MTTSystem
# =============================================================================
# Setup Dataset
# =============================================================================
topPath = r"E:\Data\c103_processing"
dataDir = r"E:\Data\c103"

dataFile = os.path.join(dataDir,"c103-1-ff-1_*_EIG16M_CdTe_{num2:0>6}.h5")
scanRange = np.concatenate(( np.array([364,368,372,376,380]), 
                             np.arange(383,406), [407] ))

# spotsDir = r"C:\Users\dpqb1\Documents\Data\c103_2024\Sample-1\c103-1-reconstruction_grains-layer-1"
spotsDir = r"E:\Data\c103\C103_1_unloaded_gripped_layer2of4\layer_1\3A_grains"
# sf.collectSpotsData(topPath,spotsDir)

spotsFile = os.path.join(topPath,"spots.npz")

spotData = np.load(spotsFile)

params = {}
params['detector'] = 'eiger'
params['imSize'] = (5000,5000)
params['yamlFile'] = os.path.join(topPath,"eiger16M_monolith_mruby_062224_FINAL.yml")
params['roiSize'] = [30,30,11]

dataFileSequence = util.getDataFileSequence(dataFile,scanRange)   

# Spot location
spotInd = 0
x = spotData['Xm'][spotInd]
y = spotData['Ym'][spotInd]
frm = int(spotData['ome_idxs'][spotInd])
eta, tth = util.xyToEtaTthRecenter(x,y,params)

# %%
# =============================================================================
# Setup Tracker
# =============================================================================
# Initial state with starting position and velocity (e.g., from the first measurement)
initial_state = {
    'com': np.zeros(3),  # Initial position from the center of mass at the first time step (only for Spot 1)
    'velocity': np.zeros(3),  # Assuming zero velocity at start
    'acceleration': np.zeros(3),  # Assuming zero acceleration for now
    'bbox': np.zeros(6)
}

# Instantiate Detector, Feature Extractor, Track Model
spot_detector = ThresholdingDetector(threshold=200)
# spot_detector = HDoGDetector()
#track_model = BasicModel(initial_state, feature_extractor=BasicFeatureExtractor())
track_model= KalmanModel(initial_state, feature_extractor= BasicFeatureExtractor(), process_noise=1e-5, measurement_noise=1e-5, dt=1, )
mht_tracker = MHTTracker(track_model=track_model,n_scan_pruning=2, plot_tree=True)
mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)

# %%
# =============================================================================
# Run Tracker
# =============================================================================
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
    
# %%
# =============================================================================
# Plot Tracks on Data
# =============================================================================
scanRange = np.arange(len(dataFileSequence))
omeRange = np.arange(11)
mht_tracker.tree.plot_all_tracks(full_data,scanRange,omeRange)


