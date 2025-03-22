# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:39:53 2025

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import time

from mtt_framework.feature_extraction import compute_principal_components
from mtt_framework.feature_extraction import compute_center_of_mass
from mtt_framework.feature_extraction import visualize_pca_on_slices

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
params['roiSize'] = [35,35,11]

dataFileSequence = util.getDataFileSequence(dataFile,scanRange)   

# Spot location
spotInd = 0
x = spotData['Xm'][spotInd]
y = spotData['Ym'][spotInd]
frm = int(spotData['ome_idxs'][spotInd])
eta, tth = util.xyToEtaTthRecenter(x,y,params)

# Determine number eta domain of ring at tth
detectDist, mmPerPixel, ff_trans, ff_tilt = util.loadYamlData(params)
r = detectDist * np.tan(tth) / mmPerPixel
outer_radius = r + (params['roiSize'][0] - 1) / 2
deta = 1/outer_radius
num_eta = len(np.arange(-np.pi,np.pi,deta))
# params['roiSize'] = [35,num_eta]

scan = 0
fname = dataFileSequence[scan]

# Precompute Interpolation Matrix
interp_params = [util.getInterpParams(tth,eta,params)]

#%% Loads data and interpolates full ring
frm_stack = []
for frm in range(2,25):
    tic = time.time()
    ring = util.loadPolarROI(fname,tth,eta,frm,params,interp_params=interp_params)
    frm_stack.append(ring)
    toc = time.time()
    # print(toc-tic)
    np.save(f'E:\\omega_frames\\frame_{frm}_scan_{scan}',ring)
    
# region = np.stack(frm_stack,axis=2)
# com = compute_center_of_mass(region)
# axes, variances = compute_principal_components(region, com)

# Visualize the PCA projection on 2D slices
# visualize_pca_on_slices(region, com, axes, variances)
    

    
    # plt.imshow(ring)
    # plt.clim(0, 100) # Set color limits between 0.2 and 0.8
    # plt.colorbar()
    # plt.show()
