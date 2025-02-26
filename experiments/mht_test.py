# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:54:34 2025

Test MHT Tracker

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mtt_framework import mht_tracker as mht
from data import synthetic_spot_data as ssd

# Generate test data
data, state_vector = ssd.generate_synthetic_data('overlapping pair', timesteps=20)

tracker = MHTTracker()

tracker.process_measurements(state_vector[0])


hypoth_tree = mht.initialize_hypothesis_tree(state_vector[0])
mht.visualize_hypothesis_tree(hypoth_tree)




m2ta1 = np.array([[1,1],[1,1]])
mht.update_hypothesis_tree(hypoth_tree, m2ta1, state_vector[1] , euclidean_cost)

mht.visualize_hypothesis_tree(hypoth_tree)
