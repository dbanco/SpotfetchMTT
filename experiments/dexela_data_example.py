# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:24 2025

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import json

# topPath = "/nfs/chess/user/dbanco/c103_processing"
# dataDir = r"/nfs/chess/raw/2024-2/id3a/miller-3528-c/c103-1-ff-1"

topPath = "/nfs/chess/aux/cycles/2025-1/id3a/shanks-3731-d/reduced_data/parameter_files"
dataDir = "/nfs/chess/raw/2025-1/id3a/shanks-3731-d/ti-2-test"

params = {}
params['detector'] = 'dexela'
params['imSize'] = (4000,6500)
# dexelas_calibrated_ruby_0504_v01.yml
params['yamlFile'] = os.path.join(topPath,"ceo2_dexela_instr_032025.yml")
params['roiSize'] = [30,30,11]
params['start_frm'] = 0

dataFile1 = os.path.join(dataDir,"2","ff","ff1_000658.h5")
dataFile2 = os.path.join(dataDir,"2","ff","ff2_000658.h5")
fnames = [dataFile1,dataFile2]

tth = 3.63*np.pi/180 #degrees
eta = np.pi/2
ring_width = 35 #pixels
ome_width = 10 #frames
detector_distance, mm_per_pixel, ff_trans, ff_tilt = util.loadYamlData(params,tth=tth,eta = 0)

rad = np.tan(tth)*detector_distance/mm_per_pixel
inner_rad = rad - (ring_width-1)/2
outer_rad = rad + (ring_width-1)/2
deta = 1/outer_rad
right_eta_vals = np.arange(-0.8*np.pi/2,0.8*np.pi/2,deta) + eta
num_right_eta = len(right_eta_vals)

x1 = inner_rad*np.cos(right_eta_vals) + params['imSize'][1]/2;
x2 = inner_rad*np.cos(right_eta_vals) + params['imSize'][1]/2;
y1 = outer_rad*np.sin(right_eta_vals) + params['imSize'][0]/2;
y2 = outer_rad*np.sin(right_eta_vals) + params['imSize'][0]/2;

img = util.loadDexImg(fnames, params, 20)

plt.figure(figsize=(50, 20))
plt.imshow(img,vmax=500)
plt.plot(x1,y1,'-')
plt.plot(x2,y2,'-')
plt.show()

params['roiSize'] = [ring_width,num_right_eta,ome_width]
Ainterp, new_center, x_cart, y_cart = util.getInterpParamsDexela(tth, eta, params)
interp_params = [Ainterp, new_center, x_cart, y_cart]
ring = util.loadPolarROI3D(fnames, tth, eta, [0,10], params, interp_params=interp_params)

# plt.figure(figsize=(50, 20))
# plt.imshow(np.sum(ring,2))

num_eta_regions = 40
total_ome = 120


