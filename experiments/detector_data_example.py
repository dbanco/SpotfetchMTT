# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:14:58 2025

Script showing how to load data from different CHESS detectors

@author: Daniel Banco
"""
import os
import numpy as np
import utilities as util

###### VD Simulated HEXD data ######
topPath = r"E:\VD_sim_processing"
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
spotInd = 0
x = spotData['Xm'][spotInd]
y = spotData['Ym'][spotInd]
frm = int(spotData['ome_idxs'][spotInd])
eta, tth = detect.xyToEtaTthRecenter(x,y,params)

fname = dataFileSequence[0]

# Loads data and interpolates ROI at tth,eta as defined by params

# 2D slice
roi = util.loadPolarROI(fname,tth,eta,frm,params)

# 3D tensor
roi3D = util.loadPolarROI3D(fnames, tth, eta, frame, params):

