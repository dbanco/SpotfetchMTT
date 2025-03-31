# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 20:20:20 2025

@author: dpqb1
"""
import sys
import os
import pickle
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import utilities as util
import numpy as np
from mtt_framework.mht_tracker import MHTTracker
from mtt_framework.state_model import KalmanModel
from mtt_system import MTTSystem
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.detection import ThresholdingDetector


track_path = "/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/tracker_states/"

for region_id in np.arange(70):
    try:
        # Load in tracker
        state_file = os.path.join(track_path,f"tracker_region_{region_id}.pkl")
        with open(state_file, "rb") as f:
            mtt = pickle.load(f)
            
        mtt.tracker.tree.visualize_hypothesis_tree()
        
        dims = [35,86,10]
        num_scans = 48
        full_data = np.zeros((num_scans,dims[0],dims[1],dims[2]))
        
        for scan in range(num_scans):
            print(f'Scan {scan}')
            # Load roi
            data = np.load(f'/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/region_files/region_{region_id}_scan_{scan}.npy')
            # Collect data series
            full_data[scan] = data
            
        scanRange = np.arange(num_scans)
        omeRange = np.arange(dims[2])
        mtt.tracker.tree.plot_all_tracks(full_data,scanRange,omeRange, vlim=None)
    except:
        print(f"No track for region {region_id}")