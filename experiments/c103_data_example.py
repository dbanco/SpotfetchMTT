# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:24 2025

@author: dpqb1
"""
import os
import numpy as np
import utilities as util

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

fname = dataFileSequence[0]

# Loads data and interpolates ROI at tth,eta as defined by params

# 2D slice
roi = util.loadPolarROI(fname,tth,eta,frm,params)

# 3D tensor
roi3D = util.loadPolarROI3D(fname,tth,eta,frm,params)
