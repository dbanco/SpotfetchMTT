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

# topPath = r"E:\Data\c103_processing"
# dataDir = r"E:\Data\c103"

topPath = "/nfs/chess/user/dbanco/c103_processing"
dataDir = "/nfs/chess/id1a3/2024-2/nygren-4125-a/nygren-series-cycle-2024-2-chessdaq"

dataFile = os.path.join(dataDir,"c103-1-ff-1_*_EIG16M_CdTe_{num2:0>6}.h5")
scanRange = np.concatenate(( np.array([364,368,372,376,380]), 
                             np.arange(383,406), [407] ))

# spotsDir = r"E:\Data\c103\C103_1_unloaded_gripped_layer2of4\layer_1\3A_grains"
# sf.collectSpotsData(topPath,spotsDir)

spotsFile = os.path.join(topPath,"spots.npz")
spotData = np.load(spotsFile)

params = {}
params['detector'] = 'eiger'
params['imSize'] = (5000,5000)
params['yamlFile'] = os.path.join(topPath,"eiger16M_monolith_mruby_062224_FINAL.yml")
params['roiSize'] = [35,35,11]

dataFileSequence = util.getDataFileSequence(dataFile,scanRange)   
outputDir = os.path.join('/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess','omega_frames',)

#%% Loads data and interpolates full ring
dfrm = (params['roiSize'][2]-1)/2
for scan in range(5):#range(len(dataFileSequence)):
    fname = dataFileSequence[scan]
    # Spot location
    for spotInd in range(2):
        x = spotData['Xm'][spotInd]
        y = spotData['Ym'][spotInd]
        frm = int(spotData['ome_idxs'][spotInd])
        eta, tth = util.xyToEtaTthRecenter(x,y,params)
        
        # Precompute Interpolation Matrix
        interp_params = [util.getInterpParams(tth,eta,params)]

        for frm in util.wrapFrame(np.arange(frm-dfrm,frm+dfrm+1)):
            frm=int(frm)
            roi = util.loadPolarROI(fname,tth,eta,int(frm),params,interp_params=interp_params)
            outFile = os.path.join(outputDir,'region_{spotInd}_frame_{int(frm)}_scan_{scan}')
            np.save(outFile,roi)
    
