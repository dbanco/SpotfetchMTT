#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities

Description:
Read and work with data from x-ray detectors

Supported detectors:
    dexela
    eiger
    eiger_sim
    
Created on: Fri Nov 15 23:00:28 2024
Author: Daniel Banco
email: dpqb10@gmail.com
"""
import numpy as np
import scipy as sp
import pandas as pd
import yaml
import h5py
import os
import time
import glob
from hexrd import imageseries
from hexrd import transforms

FRAME1 = 0
NUMFRAMES = 1440
OMEG_RANGE = 360
DEX_SHAPE = (3888, 3072)
EIG_SHAPE = (4362, 4148)

def read_yaml(file_path):
    """
    Reads a YAML file and returns the parsed data.

    Parameters:
    -----------
    file_path : str
        Path to the YAML file.

    Returns:
    --------
    dict
        Parsed YAML data.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def loadYamlData(params, tth=None, eta=None):
    """
    Loads YAML data based on detector type and parameters.

    Parameters:
    -----------
    params : dict
        Parameters including detector type and YAML file path.
    tth : float, optional
        Theta angle for the detector (used for Dexela).
    eta : float, optional
        Eta angle for the detector (used for Dexela).

    Returns:
    --------
    tuple
        Detector distance, pixel size, translation, and tilt.
    """
    yamlFile = params['yamlFile']
    detector = params['detector']

    if detector == 'dexela':
        if tth is None:
            raise ValueError("Theta (tth) must be provided for Dexela detector.")
        return loadYamlDataDexela(yamlFile, tth, eta)
    elif detector in ['eiger', 'eiger_sim']:
        return loadYamlDataEiger(yamlFile)
    else:
        raise ValueError(f"Unsupported detector type: {detector}")
        
def loadYamlDataDexela(yamlFile, tth, eta):
    """
    Loads YAML data for the Dexela detector.

    Parameters:
    -----------
    yamlFile : str
        Path to the YAML file.
    tth : float
        Theta angle for the detector.
    eta : float
        Eta angle for the detector.

    Returns:
    --------
    tuple
        Detector distance, pixel size, translation, and tilt.
    """
    yamlData = read_yaml(yamlFile)
    ff1_trans = yamlData['detectors']['ff1']['transform']['translation']
    ff2_trans = yamlData['detectors']['ff2']['transform']['translation']
    ff1_tilt = yamlData['detectors']['ff1']['transform']['tilt']
    ff2_tilt = yamlData['detectors']['ff2']['transform']['tilt']

    ff_trans = [ff1_trans[0], ff1_trans[1], ff2_trans[0], ff2_trans[1]]
    ff_tilt = [ff1_tilt, ff2_tilt]

    if abs(eta) > (np.pi / 2):
        detectDist = -ff2_trans[2]
        mmPerPixel = yamlData['detectors']['ff2']['pixels']['size'][0]
    else:
        detectDist = -ff1_trans[2]
        mmPerPixel = yamlData['detectors']['ff1']['pixels']['size'][0]

    return detectDist, mmPerPixel, ff_trans, ff_tilt

def loadYamlDataEiger(yamlFile):
    """
    Loads YAML data for the Eiger detector.

    Parameters:
    -----------
    yamlFile : str
        Path to the YAML file.

    Returns:
    --------
    tuple
        Detector distance, pixel size, translation, and tilt.
    """
    yamlData = read_yaml(yamlFile)
    trans = yamlData['detectors']['eiger']['transform']['translation']
    tilt = yamlData['detectors']['eiger']['transform']['tilt']
    detectDist = -trans[2]
    mmPerPixel = yamlData['detectors']['eiger']['pixels']['size'][0]

    return detectDist, mmPerPixel, trans, tilt

def loadImg(fnames, params, frame):
    """
    Loads an image based on the detector type.

    Parameters:
    -----------
    fnames : list
        File names for image files (one file per detector panel)
    params : dict
        Parameters including detector type.
    frame : int
        Frame index to load.

    Returns:
    --------
    ndarray
        Loaded image.
    """
    detector = params['detector']

    if detector == 'dexela':
        return loadDexImg(fnames, params, frame)
    elif detector == 'eiger':
        return loadEigerImg(fnames, params, frame)
    elif detector == 'eiger_sim':
        return loadEigerSimImg(fnames, params, frame)
    else:
        raise ValueError(f"Unsupported detector type: {detector}")
        
def loadDexImg(fnames, params, frame_i, dexSize=DEX_SHAPE):
    """
    Loads an image for the Dexela detector and aligns its panels.

    Parameters:
    -----------
    fnames : list
        List of file names (one for each panel).
    params : dict
        Parameters including image size and YAML file path.
    frame_i : int
        Frame index to load.
    dexSize : tuple, optional
        Size of each Dexela panel, default is DEX_SHAPE.

    Returns:
    --------
    ndarray
        Padded and aligned image.
    """
    with h5py.File(fnames[0], 'r') as file1, h5py.File(fnames[1], 'r') as file2:
        img1 = file1['/imageseries/images'][frame_i, :, :]
        img2 = file2['/imageseries/images'][frame_i, :, :]

    imSize = params['imSize']
    yamlFile = params['yamlFile']
    bpad = np.zeros(imSize)
    center = (imSize[0] / 2, imSize[1] / 2)

    yamlData = read_yaml(yamlFile)
    mmPerPixel = yamlData['detectors']['ff2']['pixels']['size'][0]
    ff1_trans = yamlData['detectors']['ff1']['transform']['translation']
    ff2_trans = yamlData['detectors']['ff2']['transform']['translation']

    shifts = {
        'ff1_x': int(round(ff1_trans[0] / mmPerPixel)),
        'ff1_y': int(round(ff1_trans[1] / mmPerPixel)),
        'ff2_x': int(round(ff2_trans[0] / mmPerPixel)),
        'ff2_y': int(round(ff2_trans[1] / mmPerPixel)),
    }

    # Define panel coordinates
    coords = {
        'ff1': {
            'r1': int(center[0] - dexSize[0] / 2 - shifts['ff1_y']),
            'r2': int(center[0] + dexSize[0] / 2 - shifts['ff1_y']),
            'c1': int(center[1] - dexSize[1] / 2 + shifts['ff1_x']),
            'c2': int(center[1] + dexSize[1] / 2 + shifts['ff1_x']),
        },
        'ff2': {
            'r1': int(center[0] - dexSize[0] / 2 - shifts['ff2_y']),
            'r2': int(center[0] + dexSize[0] / 2 - shifts['ff2_y']),
            'c1': int(center[1] - dexSize[1] / 2 + shifts['ff2_x']),
            'c2': int(center[1] + dexSize[1] / 2 + shifts['ff2_x']),
        }
    }

    # Assign flipped images to padded array
    bpad[coords['ff2']['r1']:coords['ff2']['r2'], coords['ff2']['c1']:coords['ff2']['c2']] = np.flipud(img2)
    bpad[coords['ff1']['r1']:coords['ff1']['r2'], coords['ff1']['c1']:coords['ff1']['c2']] = np.fliplr(img1)

    return bpad

def loadEigerImg(fnames, params, frame, detectSize=EIG_SHAPE):
    """
    Loads and aligns an image for the Eiger detector.

    Parameters:
    -----------
    fnames : str
        File name of the Eiger image series.
    params : dict
        Parameters including image size and YAML file path.
    frame : int
        Frame index to load.
    detectSize : tuple, optional
        Size of the Eiger detector, default is EIG_SHAPE.

    Returns:
    --------
    ndarray
        Padded and aligned image.
    """
    # 0. Load params, YAML data
    imSize = params['imSize']
    yamlFile = params['yamlFile']
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataEiger(yamlFile)

    ims = imageseries.open(fnames, format='eiger-stream-v1')
    img = ims[frame, :, :].copy()
    img[img > 4294000000] = 0

    # Pad image
    bpad = np.zeros(imSize)
    center = (imSize[0] / 2, imSize[1] / 2)

    # Shift each panel
    ff1_tx = ff_trans[0]
    ff1_ty = ff_trans[1]
    ff1_xshift = int(round(ff1_tx / mmPerPixel))
    ff1_yshift = int(round(ff1_ty / mmPerPixel))

    # Negative sign on y shift because rows increase downwards
    ff1r1 = int(center[0] - detectSize[0] / 2 - ff1_yshift)
    ff1r2 = int(center[0] + detectSize[0] / 2 - ff1_yshift)

    ff1c1 = int(center[1] - detectSize[1] / 2 + ff1_xshift)
    ff1c2 = int(center[1] + detectSize[1] / 2 + ff1_xshift)

    bpad[ff1r1:ff1r2, ff1c1:ff1c2] = img

    return bpad

def loadEigerSimImg(fnames, params, frame_i, detectSize=EIG_SHAPE):
    """
    Loads and aligns a simulated image for the Eiger detector.

    Parameters:
    -----------
    fnames : str
        File name of the simulated data (.npz format).
    params : dict
        Parameters including image size and YAML file path.
    frame_i : int
        Frame index to load.
    detectSize : tuple, optional
        Size of the Eiger detector, default is EIG_SHAPE.

    Returns:
    --------
    ndarray
        Padded and aligned image.
    """
    simData = np.load(fnames)
    shp = simData['shape']
    img = np.zeros((shp[0], shp[1]))

    rowD = simData[f'{frame_i}_row']
    colD = simData[f'{frame_i}_col']
    datD = simData[f'{frame_i}_data']

    for i in range(len(rowD)):
        img[rowD[i], colD[i]] = datD[i]

    imSize = params['imSize']
    yamlFile = params['yamlFile']

    # Pad image
    bpad = np.zeros(imSize)
    center = (imSize[0] / 2, imSize[1] / 2)

    # Shift each panel
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataEiger(yamlFile)
    ff1_tx = ff_trans[0]
    ff1_ty = ff_trans[1]
    ff1_xshift = int(round(ff1_tx / mmPerPixel))
    ff1_yshift = int(round(ff1_ty / mmPerPixel))

    # Negative sign on y shift because rows increase downwards
    ff1r1 = int(center[0] - detectSize[0] / 2 - ff1_yshift)
    ff1r2 = int(center[0] + detectSize[0] / 2 - ff1_yshift)

    ff1c1 = int(center[1] - detectSize[1] / 2 + ff1_xshift)
    ff1c2 = int(center[1] + detectSize[1] / 2 + ff1_xshift)

    bpad[ff1r1:ff1r2, ff1c1:ff1c2] = img

    return bpad

def getInterpParams(tth, eta, params):
    """
    Retrieves interpolation parameters for the detector.

    Parameters:
    -----------
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    params : dict
        Parameters including detector type and YAML file path.

    Returns:
    --------
    tuple
        Interpolation matrix, new center, x_cart, and y_cart coordinates.
    """
    if params['detector'] == 'dexela':
        return getInterpParamsDexela(tth, eta, params)
    elif params['detector'] == 'eiger':
        return getInterpParamsEiger(tth, eta, params)
    elif params['detector'] == 'eiger_sim':
        return getInterpParamsEiger(tth, eta, params)

def getInterpParamsDexela(tth, eta, params):
    """
    Retrieves interpolation parameters for the Dexela detector.

    Parameters:
    -----------
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    params : dict
        Parameters including detector type and YAML file path.

    Returns:
    --------
    tuple
        Interpolation matrix, new center, x_cart, and y_cart coordinates.
    """
    yamlFile = params['yamlFile']
    roiSize = params['roiSize']
    imSize = params['imSize']

    center = (imSize[0] / 2, imSize[1] / 2)
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataDexela(yamlFile, tth, eta)

    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)
    x_cart, y_cart = fetchCartesian(rad_dom, eta_dom, center)
    ff1_pix, ff2_pix = panelPixelsDex(ff_trans, mmPerPixel, imSize)

    
    roiShape = getROIshapeDex(x_cart, y_cart, ff1_pix, ff2_pix, center)
    
    new_center = np.array([center[0] - y_cart[0], center[1] - x_cart[0]])
    Ainterp = bilinearInterpMatrix(roiShape, rad_dom, eta_dom, new_center, detectDist)

    return Ainterp, new_center, x_cart, y_cart

def getInterpParamsEiger(tth, eta, params):
    """
    Retrieves interpolation parameters for the Eiger detector.

    Parameters:
    -----------
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    params : dict
        Parameters including YAML file path, ROI size, and image size.

    Returns:
    --------
    tuple
        Interpolation matrix, new center, x_cart, and y_cart coordinates.
    """
    yamlFile = params['yamlFile']
    roiSize = params['roiSize']
    imSize = params['imSize']

    center = ((imSize[0] - 1) / 2, (imSize[1] - 1) / 2)
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataEiger(yamlFile)

    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)

    # Get interpolation matrix
    x_cart, y_cart = fetchCartesian(rad_dom, eta_dom, center)
    ff_pix = panelPixelsEiger(ff_trans, mmPerPixel, imSize)

    new_center = np.array([center[0] - y_cart[0], center[1] - x_cart[0]])
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff_pix)
    roiShape = [y_pan[1]-y_pan[0],x_pan[1]-x_pan[0]]

    Ainterp = bilinearInterpMatrix(roiShape, rad_dom, eta_dom, new_center, detectDist)

    return Ainterp, new_center, x_cart, y_cart

def loadROI(dataPath, scan, frame, etaRoi, tthRoi, params):
    """
    Loads the region of interest (ROI) for the given detector.

    Parameters:
    -----------
    dataPath : str
        Path to the data.
    scan : int
        Scan number.
    frame : int
        Frame index.
    etaRoi : float
        Eta range of interest.
    tthRoi : float
        Two-theta range of interest.
    params : dict
        Detector parameters.

    Returns:
    --------
    ndarray
        Loaded polar ROI.
    """
    if os.path.isdir(dataPath):
        fnames = timeToFile(scan, dataPath)
        isFile = False
    else:
        fnames = dataPath
        isFile = True
        if params['detector'] == 'eiger_sim':
            fnames = fnames.replace('*', '{scan}').format(scan=scan)
            roi = loadPolarROI(fnames, tthRoi, etaRoi, frame, params)
            return roi

    # Load ROI
    if isFile:
        template = dataPath.format(num2=scan)
        fnames = glob.glob(template)
        roi = loadPolarROI(fnames[0], tthRoi, etaRoi, frame, params)
    else:
        roi = loadPolarROI(fnames, tthRoi, etaRoi, frame, params)

    return roi

def loadPolarROI(fnames, tth, eta, frame, params, interp_params=None):
    """
    Loads the polar ROI for the specified detector type.

    Parameters:
    -----------
    fnames : str or list
        File name(s) of the image data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Frame index.
    params : dict
        Detector parameters.

    Returns:
    --------
    ndarray
        Loaded polar ROI.
    """
    if params['detector'] == 'dexela':
        roi = loadDexPolarRoi(fnames, tth, eta, frame, params)
    elif params['detector'] == 'eiger':
        roi = loadEigerPolarRoi(fnames, tth, eta, frame, params)
    elif params['detector'] == 'eiger_sim':
        roi = loadEigerPolarRoi(fnames, tth, eta, frame, params)
    return roi

def loadPolarROI3D(fnames, tth, eta, frame, params, interp_params=None):
    """
    Loads a 3D polar ROI for the specified detector type.

    Parameters:
    -----------
    fnames : str or list
        File name(s) of the image data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Central frame index.
    params : dict
        Detector parameters.

    Returns:
    --------
    ndarray
        Loaded 3D polar ROI.
    """
    if params['detector'] == 'dexela':
        roi3D = loadDexPolarRoi3D(fnames, tth, eta, frame, params, interp_params)
    elif params['detector'] == 'eiger':
        roi3D = loadEigerPolarRoi3D(fnames, tth, eta, frame, params, interp_params)
    elif params['detector'] == 'eiger_sim':
        roi3D = loadEigerPolarRoi3D(fnames, tth, eta, frame, params, interp_params)
    return roi3D

def loadDexPolarRoi(fnames, tth, eta, frame, params, interp_params=None):
    """
    Loads the polar ROI for the Dexela detector.

    Parameters:
    -----------
    fnames : str or list
        File name(s) of the image data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Frame index.
    params : dict
        Detector parameters.

    Returns:
    --------
    ndarray
        Loaded polar ROI.
    """
    # Load parameters and YAML data
    yamlFile = params['yamlFile']
    roiSize = params['roiSize']
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataDexela(yamlFile, tth, eta)

    # Construct radial and eta domains
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)

    # Construct interpolation matrix
    if interp_params == None:
        Ainterp, new_center, x_cart, y_cart = getInterpParamsDexela(tth, eta, params)
    else:
        Ainterp, new_center, x_cart, y_cart = interp_params

    # Load Cartesian ROI pixels
    ff1_pix, ff2_pix = panelPixelsDex(ff_trans, mmPerPixel,imSize=params['imSize'])
    roi = loadDexPanelROI(x_cart, y_cart, ff1_pix, ff2_pix, fnames, frame, params)

    # Apply interpolation matrix to Cartesian pixels to get polar values
    roi_polar_vec = Ainterp.dot(roi.flatten())

    # Reshape and return ROI
    roi_polar = np.reshape(roi_polar_vec, (len(rad_dom), len(eta_dom)))

    return roi_polar

def loadDexPolarRoi3D(fnames, tth, eta, frames, params, interp_params=None):
    """
    Loads a 3D polar ROI for the Dexela detector.

    Parameters:
    -----------
    fnames : str or list
        File name(s) of the image data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Central frame index.
    params : dict
        Detector parameters.

    Returns:
    --------
    ndarray
        Loaded 3D polar ROI.
    """
    # Load parameters and YAML data
    yamlFile = params['yamlFile']
    roiSize = params['roiSize']
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlDataDexela(yamlFile, tth, eta)
    frmRange = np.arange(frames[0], frames[1])
    
    # Construct radial and eta domains
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)

    # Construct interpolation matrix
    if interp_params == None:
        Ainterp, new_center, x_cart, y_cart = getInterpParamsDexela(tth, eta, params)
    else:
        Ainterp = interp_params[0]
        new_center = interp_params[1]
        x_cart = interp_params[2]
        y_cart = interp_params[3]
        
    # Load Cartesian ROI pixels for each frame
    ff1_pix, ff2_pix = panelPixelsDex(ff_trans, mmPerPixel,imSize=params['imSize'])
    roi3D = np.zeros(roiSize)
    for i, frm in enumerate(frmRange):
        roi = loadDexPanelROI(x_cart, y_cart, ff1_pix, ff2_pix, fnames, int(frm), params)
        roi_polar_vec = Ainterp.dot(roi.flatten())
        roi_polar = np.reshape(roi_polar_vec, roiSize[:2])
        roi3D[:, :, i] = roi_polar

    return roi3D

def loadEigerPolarRoi(fname, tth, eta, frame, params, interp_params=None):
    """
    Loads a polar region of interest (ROI) for the Eiger detector.

    Parameters:
    -----------
    fname : str
        File name of the data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Frame index to load.
    params : dict
        Parameters including ROI size, image size, and YAML file path.

    Returns:
    --------
    ndarray
        Reshaped polar ROI.
    """
    # 0. Load params, YAML data
    roiSize = params['roiSize']
    imSize = params['imSize']
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlData(params)
    
    # 1. Construct rad, eta domain
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)
    
    # 2. Construct interpolation matrix
    if interp_params == None:
        Ainterp, new_center, x_cart, y_cart = getInterpParamsEiger(tth, eta, params)
    else:
        Ainterp = interp_params[0]
        new_center = interp_params[1]
        x_cart = interp_params[2]
        y_cart = interp_params[3]
        
    # 3. Load needed Cartesian ROI pixels
    ff1_pix = panelPixelsEiger(ff_trans, mmPerPixel, imSize)
    if params['detector'] == 'eiger':
        roi = loadEigerPanelROI(x_cart, y_cart, ff1_pix, fname, frame)
    elif params['detector'] == 'eiger_sim':
        roi = loadEigerSimPanelROI(x_cart, y_cart, ff1_pix, fname, frame)
    
    tic = time.time()
    # 4. Apply interpolation matrix to Cartesian pixels get Polar values
    roi_polar_vec = Ainterp.dot(roi.flatten())
    
    # 5. Reshape and output roi
    roi_polar = np.reshape(roi_polar_vec, (len(rad_dom), len(eta_dom)))
    toc = time.time()
    print(toc-tic)

    return roi_polar

def loadEigerPolarRoi3D(fname,tth,eta,frame,params,interp_params=None):
    """
    Loads a 3D polar region of interest (ROI) for the Eiger detector.

    Parameters:
    -----------
    fname : str
        File name of the data.
    tth : float
        Two-theta value.
    eta : float
        Eta value.
    frame : int
        Frame index to load.
    params : dict
        Parameters including ROI size, image size, and YAML file path.

    Returns:
    --------
    ndarray
        3D polar ROI.
    """
    # 0. Load params, YAML data
    roiSize = params['roiSize']
    imSize = params['imSize']
    dome = (roiSize[2]-1)/2
    frmRange = wrapFrame(np.arange(frame-dome,frame+dome+1),frm0=params['start_frm'])
    
    detectDist, mmPerPixel, ff_trans, ff_tilt = loadYamlData(params)
    
    # 1. Construct rad, eta domain
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tth, eta, roiSize)
    
    # 2. Construct interpolation matrix
    if interp_params == None:
        Ainterp,new_center,x_cart,y_cart = getInterpParamsEiger(tth,eta,params)
    else:
        Ainterp, new_center, x_cart, y_cart = interp_params
        
    ff1_pix = panelPixelsEiger(ff_trans,mmPerPixel,imSize)

    # 3. Load needed Cartesian ROI pixels
    if params['detector'] == 'eiger':
        roi = loadEigerPanelROI3D(x_cart, y_cart, ff1_pix, fname, frmRange)
    elif params['detector'] == 'eiger_sim':
        roi = loadEigerSimPanelROI3D(x_cart, y_cart, ff1_pix, fname, frmRange)
    # 4. Apply interpolation matrix to Cartesian pixels get Polar values
    roi_polar_vec = Ainterp.dot(roi.reshape([roi.shape[0]*roi.shape[1],roi.shape[2]]))
    # 5. Reshape and output roi
    roi_polar = np.reshape(roi_polar_vec, roiSize)
    
    return roi_polar

# def loadDexPanelROIOLD(x_cart, y_cart, ff1_pix, ff2_pix, fnames, frame, params, interp_params=None, dexShape=DEX_SHAPE):
#     """
#     Loads a Region of Interest (ROI) for the Dexela detector in Cartesian coordinates.

#     Parameters:
#     -----------
#     x_cart : list or ndarray
#         X-coordinates of the ROI in the Cartesian system.
#     y_cart : list or ndarray
#         Y-coordinates of the ROI in the Cartesian system.
#     ff1_pix : ndarray
#         Flat-field parameters for the first panel (right panel).
#     ff2_pix : ndarray
#         Flat-field parameters for the second panel (left panel).
#     fnames : list
#         List of file names containing panel data.
#     frame : int
#         Frame index to load.
#     params : dict
#         Dictionary of detector parameters, including image size.
#     dexShape : tuple, optional
#         Shape of the Dexela detector, default is DEX_SHAPE.

#     Returns:
#     --------
#     ndarray
#         Loaded ROI image from the specified panel.
#     """
#     # Extract image size and calculate the center
#     imSize = params['imSize']
#     center = (imSize[0] / 2, imSize[1] / 2)

#     # Ensure ROI boundaries do not exceed panel limits
#     if x_cart[0] < ff2_pix[0]: x_cart[0] = ff2_pix[0]
#     if x_cart[1] > ff1_pix[1]: x_cart[1] = ff1_pix[1]
#     if y_cart[0] < ff2_pix[2]: y_cart[0] = ff2_pix[2]
#     if y_cart[1] > ff2_pix[3]: y_cart[1] = ff2_pix[3]

#     # Determine which panel the ROI belongs to
#     if x_cart[0] < center[1]:  # Left panel
#         # Adjust for panel offsets
#         x_pan = x_cart - ff2_pix[0]
#         y_pan = y_cart - ff2_pix[2]
#         # Account for vertical flipping
#         midLine = (dexShape[0] - 1) / 2
#         flip0 = y_pan[0] + 2 * (midLine - y_pan[0])
#         flip1 = y_pan[1] + 2 * (midLine - y_pan[1])
#         y_pan[0] = min(flip0, flip1)
#         y_pan[1] = max(flip0, flip1)
#         # Load and flip the image vertically
#         with h5py.File(fnames[1], 'r') as file:
#             img = file['/imageseries/images'][frame, y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]]
#             img = np.flipud(img)
#     elif x_cart[0] > center[1]:  # Right panel
#         # Adjust for panel offsets
#         x_pan = x_cart - ff1_pix[0]
#         y_pan = y_cart - ff1_pix[2]
#         # Account for horizontal flipping
#         midLine = (dexShape[1] - 1) / 2
#         flip0 = x_pan[0] + 2 * (midLine - x_pan[0])
#         flip1 = x_pan[1] + 2 * (midLine - x_pan[1])
#         x_pan[0] = min(flip0, flip1)
#         x_pan[1] = max(flip0, flip1)
#         # Load and flip the image horizontally
#         with h5py.File(fnames[0], 'r') as file:
#             print(file['/imageseries/images'])
#             img = file['/imageseries/images'][frame, y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]]
#             img = np.fliplr(img)

#     return img

def loadDexPanelROI(x_cart, y_cart, ff1_pix, ff2_pix, fnames, frame, params, dexShape=DEX_SHAPE):
    """
    Load ROI from Dexela detector with support for missing regions (filled with zeros).
    The ROI is defined in global coordinates.
    """
    x0, x1 = int(np.floor(x_cart[0])), int(np.ceil(x_cart[1]))
    y0, y1 = int(np.floor(y_cart[0])), int(np.ceil(y_cart[1]))
    roi_width = x1 - x0
    roi_height = y1 - y0

    # Preallocate full output ROI image
    full_img = np.zeros((roi_height, roi_width), dtype=np.float32)

    # Panel bounding boxes in global coordinates: [xmin, xmax, ymin, ymax]
    panels = [
        {'file': fnames[0], 'bbox': ff1_pix, 'flip': 'horizontal'},  # Right panel
        {'file': fnames[1], 'bbox': ff2_pix, 'flip': 'vertical'},    # Left panel
    ]

    for panel in panels:
        xmin, xmax, ymin, ymax = map(int, panel['bbox'])

        # Find intersection with ROI
        ix0 = max(x0, xmin)
        ix1 = min(x1, xmax)
        iy0 = max(y0, ymin)
        iy1 = min(y1, ymax)

        if ix0 >= ix1 or iy0 >= iy1:
            continue  # No overlap with this panel

        # Indices in the panel image
        px0 = ix0 - xmin
        px1 = ix1 - xmin
        py0 = iy0 - ymin
        py1 = iy1 - ymin

        # Indices in the ROI canvas
        cx0 = ix0 - x0
        cx1 = ix1 - x0
        cy0 = iy0 - y0
        cy1 = iy1 - y0

        with h5py.File(panel['file'], 'r') as f:
            img = f['/imageseries/images'][frame, py0:py1, px0:px1]

        # Apply flip if needed
        if panel['flip'] == 'horizontal':
            img = np.fliplr(img)
        elif panel['flip'] == 'vertical':
            img = np.flipud(img)

        # Insert into the full ROI canvas
        full_img[cy0:cy1, cx0:cx1] = img

    return full_img

def loadEigerPanelROI(x_cart, y_cart, ff1_pix, fname, frame):
    """
    Loads a Region of Interest (ROI) for the Eiger detector in Cartesian coordinates.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the image series.
    frame : int
        Frame index to load.

    Returns:
    --------
    ndarray
        Loaded ROI image.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load image data
    ims = imageseries.open(fname, format='eiger-stream-v1')
    img = ims[frame, y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()

    # Mask invalid pixel values
    img[img > 4294000000] = 0
    return img

def loadEigerPanelROI3D(x_cart, y_cart, ff1_pix, fname, frameRange):
    """
    Loads a Region of Interest (ROI) for the Eiger detector in Cartesian coordinates.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the image series.
    frame : int
        Frame index to load.

    Returns:
    --------
    ndarray
        Loaded ROI image.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load image data
    img = np.zeros((y_cart[1]-y_cart[0],x_cart[1]-x_cart[0],len(frameRange)))
    ims = imageseries.open(fname, format='eiger-stream-v1')
    for i, frm in enumerate(frameRange):
        img[:,:,i] = ims[int(frm), y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()

    # Mask invalid pixel values
    img[img > 4294000000] = 0
    return img

def loadEigerSimPanelROI(x_cart, y_cart, ff1_pix, fname, frame):
    """
    Loads a simulated Region of Interest (ROI) for the Eiger detector.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the simulation data (NumPy file).
    frame : int
        Frame index to load.

    Returns:
    --------
    ndarray
        Loaded simulated ROI image.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load simulation data
    simData = np.load(fname)
    shp = simData['shape']
    imgFull = np.zeros((shp[0], shp[1]))

    # Load data for the specific frame
    frame = int(frame - 2)
    rowD = simData[f'{frame}_row']
    colD = simData[f'{frame}_col']
    datD = simData[f'{frame}_data']

    # Populate the full image array
    for i in range(len(rowD)):
        imgFull[rowD[i], colD[i]] = datD[i]

    # Extract and return the ROI
    img = imgFull[y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()
    return img

def loadEigerSimPanelROI3D(x_cart, y_cart, ff1_pix, fname, frameRange):
    """
    Loads a simulated Region of Interest (ROI) for the Eiger detector.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the simulation data (NumPy file).
    frame : int
        Frame index to load.

    Returns:
    --------
    ndarray
        Loaded simulated ROI image.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load simulation data
    simData = np.load(fname)
    shp = simData['shape']
    
    img = np.zeros((y_cart[1]-y_cart[0],x_cart[1]-x_cart[0],len(frameRange)))
    for i, frm in enumerate(frameRange):
        # Load data for the specific frame
        frame = int(frm - 2)
        rowD = simData[f'{frame}_row']
        colD = simData[f'{frame}_col']
        datD = simData[f'{frame}_data']
           
        # Populate the full image array
        imgFull = np.zeros(shp)
        for j in range(len(rowD)):
            imgFull[rowD[j], colD[j]] = datD[j]
    
        # Extract and return the ROI
        img[:,:,i] = imgFull[y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()
    return img

def loadEigerPanelROIArray(x_cart, y_cart, ff1_pix, fname, frames):
    """
    Loads a series of Regions of Interest (ROIs) for the Eiger detector across multiple frames.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the image series.
    frames : tuple
        Start and end frame indices (inclusive).

    Returns:
    --------
    ndarray
        3D array containing ROIs for the specified frames.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load image data
    ims = imageseries.open(fname, format='eiger-stream-v1')
    imgArray = np.zeros((frames[1] - frames[0], y_pan[1] - y_pan[0], x_pan[1] - x_pan[0]))

    # Extract ROI for each frame
    for i in np.arange(frames[0], frames[1]):
        imgArray[i - frames[0], :, :] = ims[i, y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()

    # Mask invalid pixel values
    imgArray[imgArray > 4294000000] = 0
    return imgArray

def loadEigerSimPanelROIArray(x_cart, y_cart, ff1_pix, fname, frames):
    """
    Loads a series of simulated Regions of Interest (ROIs) for the Eiger detector across multiple frames.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : ndarray
        Flat-field parameters for the panel.
    fname : str
        File name of the simulation data (NumPy file).
    frames : tuple
        Start and end frame indices (inclusive).

    Returns:
    --------
    ndarray
        3D array containing simulated ROIs for the specified frames.
    """
    # Compute panel-specific coordinates
    x_pan, y_pan = getEigerPixels(x_cart, y_cart, ff1_pix)

    # Load simulation data
    simData = np.load(fname)
    shp = simData['shape']
    imgFull = np.zeros((frames[1] - frames[0], shp[0], shp[1]))

    # Extract ROI for each frame in the specified range
    for frm in np.arange(frames[0], frames[1]):
        frm = frm - 2  # Adjust frame index
        rowD = simData[f'{frm}_row']
        colD = simData[f'{frm}_col']
        datD = simData[f'{frm}_data']

        # Populate the full image array for the frame
        for i in range(len(rowD)):
            imgFull[2 + frm - frames[0], rowD[i], colD[i]] = datD[i]

    # Extract and return the ROI for all frames
    return imgFull[:, y_pan[0]:y_pan[1], x_pan[0]:x_pan[1]].copy()

def getROIshapeDex(x_cart, y_cart, ff1_pix, ff2_pix, center, dexShape=DEX_SHAPE):
    """
    Calculates the shape of the ROI for the Dexela detector based on Cartesian coordinates.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : list or ndarray
        Flat-field parameters for the primary panel.
    ff2_pix : list or ndarray
        Flat-field parameters for the secondary panel.
    center : tuple
        Image center coordinates.
    dexShape : tuple, optional
        Shape of the Dexela detector panel (default is DEX_SHAPE).

    Returns:
    --------
    tuple
        Shape of the ROI (rows, columns).
    """
    # Ensure ROI boundaries do not exceed panel limits
    if x_cart[0] < ff2_pix[0]: x_cart[0] = ff2_pix[0]
    if x_cart[1] > ff1_pix[1]: x_cart[1] = ff1_pix[1]
    if y_cart[0] < ff2_pix[2]: y_cart[0] = ff2_pix[2]
    if y_cart[1] > ff2_pix[3]: y_cart[1] = ff2_pix[3]

    # Determine the panel and process accordingly
    if x_cart[0] < center[1]:  # Panel 2
        x_pan = x_cart - ff2_pix[0]
        y_pan = y_cart - ff2_pix[2]

        # Account for vertical flipping
        midLine = (dexShape[0] - 1) / 2
        flip0 = y_pan[0] + 2 * (midLine - y_pan[0])
        flip1 = y_pan[1] + 2 * (midLine - y_pan[1])
        y_pan[0] = min(flip0, flip1)
        y_pan[1] = max(flip0, flip1)
    elif x_cart[0] > center[1]:  # Panel 1
        x_pan = x_cart - ff1_pix[0]
        y_pan = y_cart - ff1_pix[2]

        # Account for horizontal flipping
        midLine = (dexShape[1] - 1) / 2
        flip0 = x_pan[0] + 2 * (midLine - x_pan[0])
        flip1 = x_pan[1] + 2 * (midLine - x_pan[1])
        x_pan[0] = min(flip0, flip1)
        x_pan[1] = max(flip0, flip1)

    # Calculate and return the shape of the ROI
    return y_pan[1] - y_pan[0], x_pan[1] - x_pan[0]

def getEigerPixels(x_cart, y_cart, ff1_pix, eigShape=EIG_SHAPE):
    """
    Computes the pixel coordinates of the ROI for the Eiger detector.

    Parameters:
    -----------
    x_cart : list or ndarray
        X-coordinates of the ROI in the Cartesian system.
    y_cart : list or ndarray
        Y-coordinates of the ROI in the Cartesian system.
    ff1_pix : list or ndarray
        Flat-field parameters for the panel.
    eigShape : tuple, optional
        Shape of the Eiger detector panel (default is (4362, 4148)).

    Returns:
    --------
    tuple
        X and Y coordinates of the ROI in the panel's coordinate system.
    """
    # Ensure coordinates are within bounds
    if x_cart[0] < ff1_pix[0]: 
        x_cart[0] = ff1_pix[0]
    if x_cart[1] > ff1_pix[1]: 
        x_cart[1] = ff1_pix[1]
    if y_cart[0] < ff1_pix[2]: 
        y_cart[0] = ff1_pix[2]
    if y_cart[1] > ff1_pix[3]: 
        y_cart[1] = ff1_pix[3]

    # Transform to panel-specific coordinates
    x_pan = x_cart - ff1_pix[0]
    y_pan = y_cart - ff1_pix[2]
    return x_pan, y_pan

def bilinearInterpMatrix(roiShape, rad_dom, eta_dom, center, detectDist):
    """
    Constructs the bilinear interpolation matrix for mapping Cartesian pixels to polar coordinates.

    Parameters:
    -----------
    roiShape : tuple
        Shape of the ROI (rows, columns).
    rad_dom : ndarray
        Radial domain values.
    eta_dom : ndarray
        Azimuthal domain values.
    center : tuple
        Image center coordinates.
    detectDist : float
        Detector distance.

    Returns:
    --------
    scipy.sparse.coo_array
        Sparse matrix for bilinear interpolation.
    """
    
    out_rows = len(rad_dom)
    out_cols = len(eta_dom)
    in_rows = roiShape[0]
    in_cols = roiShape[1]
    
    row_indices = []
    col_indices = []
    values = []
    
    k = 0  # Row index in the interpolation matrix

    # Loop through radial and azimuthal domains
    for r in rad_dom:
        for eta in eta_dom:
            # Convert polar to Cartesian coordinates
            x = r * np.cos(eta) + center[1]
            y = -r * np.sin(eta) + center[0]

            # Determine surrounding pixel coordinates
            x1 = np.floor(x)
            x2 = np.ceil(x)
            y1 = np.floor(y)
            y2 = np.ceil(y)
            
            dx1 = x - x1
            dy1 = y - y1
            dx2 = x2 - x
            dy2 = y2 - y

            # Flattened indices in input image
            idx00 = y1 * in_cols + x1
            idx01 = y2 * in_cols + x1
            idx10 = y1 * in_cols + x2
            idx11 = y2 * in_cols + x2

            # Store nonzero values
            row_indices.extend([k] * 4)
            col_indices.extend([idx00, idx01, idx10, idx11])
            values.extend([dx1*dy1,dx2*dy1,dx1*dy2,dx2*dy2])
            k += 1

    Ainterp = sp.sparse.csr_array((values, (row_indices, col_indices)), 
                                  shape=(out_rows*out_cols, in_rows*in_cols))
    
    return Ainterp

def panelPixelsDex(ff_trans, mmPerPixel, imSize, dexShape=DEX_SHAPE):
    """
    Computes pixel boundaries for Dexela detector panels based on flat-field transformations.

    Parameters:
    -----------
    ff_trans : list or ndarray
        Flat-field translation parameters for each panel.
    mmPerPixel : float
        Millimeters per pixel.
    imSize : tuple, optional
        Size of the full detector image (default is (4888, 7300)).
    dexShape : tuple, optional
        Shape of the Dexela detector panel (default is (3888, 3072)).

    Returns:
    --------
    tuple
        Pixel boundaries for panels 1 and 2.
    """
    center = (imSize[0] / 2, imSize[1] / 2)

    # Calculate pixel shifts for each panel
    ff1_xshift = int(round(ff_trans[0] / mmPerPixel))
    ff1_yshift = int(round(ff_trans[1] / mmPerPixel))
    ff2_xshift = int(round(ff_trans[2] / mmPerPixel))
    ff2_yshift = int(round(ff_trans[3] / mmPerPixel))

    # Compute panel boundaries
    ff1_pixels = [
        int(center[1] - dexShape[1] / 2 + ff1_xshift),  # Column start
        int(center[1] + dexShape[1] / 2 + ff1_xshift),  # Column end
        int(center[0] - dexShape[0] / 2 - ff1_yshift),  # Row start
        int(center[0] + dexShape[0] / 2 - ff1_yshift),  # Row end
    ]
    ff2_pixels = [
        int(center[1] - dexShape[1] / 2 + ff2_xshift),  # Column start
        int(center[1] + dexShape[1] / 2 + ff2_xshift),  # Column end
        int(center[0] - dexShape[0] / 2 - ff2_yshift),  # Row start
        int(center[0] + dexShape[0] / 2 - ff2_yshift),  # Row end
    ]

    return ff1_pixels, ff2_pixels

def panelPixelsEiger(ff_trans, mmPerPixel, imSize=(5000, 5000), detectShape=EIG_SHAPE):
    """
    Computes pixel ranges for an Eiger detector panel after translation shifts.

    Parameters:
    -----------
    ff_trans : list or tuple
        Translation shifts [x_shift, y_shift] in millimeters.
    mmPerPixel : float
        Conversion factor from millimeters to pixels.
    imSize : tuple, optional
        Size of the full detector image (height, width).
    detectShape : tuple, optional
        Shape of the detector panel (height, width).

    Returns:
    --------
    ff1_pixels : list
        Pixel boundaries [col_start, col_end, row_start, row_end] for the panel.
    """
    # Determine the center of the image in pixel coordinates
    center = (imSize[0] / 2, imSize[1] / 2)

    # Translate shifts from millimeters to pixels
    ff1_tx = ff_trans[0]
    ff1_ty = ff_trans[1]
    ff1_xshift = int(round(ff1_tx / mmPerPixel))
    ff1_yshift = int(round(ff1_ty / mmPerPixel))

    # Calculate row and column boundaries based on the panel shape and translation
    ff1r1 = int(center[0] - detectShape[0] / 2 - ff1_yshift)
    ff1r2 = int(center[0] + detectShape[0] / 2 - ff1_yshift)
    ff1c1 = int(center[1] - detectShape[1] / 2 + ff1_xshift)
    ff1c2 = int(center[1] + detectShape[1] / 2 + ff1_xshift)

    # Return the calculated pixel boundaries
    ff1_pixels = [ff1c1, ff1c2, ff1r1, ff1r2]
    return ff1_pixels

def polarDomain(detectDist, mmPerPixel, tth, eta, roi_size):
    """
    Constructs radial and azimuthal domains in polar coordinates.

    Parameters:
    -----------
    detectDist : float
        Detector distance from the sample in millimeters.
    mmPerPixel : float
        Conversion factor from millimeters to pixels.
    tth : float
        Two-theta angle in radians.
    eta : float
        Eta angle in radians.
    roi_size : tuple
        Size of the ROI (radial size, angular size).

    Returns:
    --------
    rad_domain : ndarray
        Array of radial positions in pixels.
    eta_domain : ndarray
        Array of azimuthal angles in radians.
    """
    # Calculate the central radial distance in pixels based on two-theta
    r = detectDist * np.tan(tth) / mmPerPixel

    # Define the radial domain (range) based on ROI size
    r1 = r - (roi_size[0] - 1) / 2
    r2 = r + (roi_size[0] - 1) / 2
    rad_domain = np.arange(r1, r2 + 1, 1)

    # Define the azimuthal domain based on ROI size and radial limits
    deta = 1 / r2  # Small angle approximation
    eta1 = eta - roi_size[1] / 2 * deta
    eta2 = eta + roi_size[1] / 2 * deta
    eta_domain = np.linspace(eta1, eta2, roi_size[1])

    return rad_domain, eta_domain

def fetchCartesian(rad_dom, eta_dom, center):
    """
    Converts polar domain boundaries into Cartesian bounds for a given center.

    Parameters:
    -----------
    rad_dom : ndarray
        Array of radial positions.
    eta_dom : ndarray
        Array of azimuthal angles in radians.
    center : tuple
        Center of the image (row, column) in pixel coordinates.

    Returns:
    --------
    x_cart : ndarray
        Minimum and maximum x-coordinates in pixels.
    y_cart : ndarray
        Minimum and maximum y-coordinates in pixels.
    """
    # Compute x-coordinates for both radial boundaries at all eta values
    rad1 = rad_dom[0] * np.cos(eta_dom)
    rad2 = rad_dom[-1] * np.cos(eta_dom)
    x_min1 = np.min(rad1 + center[1])
    x_max1 = np.max(rad2 + center[1])
    x_max2 = np.max(rad1 + center[1])
    x_min2 = np.min(rad2 + center[1])

    # Find overall x-coordinate bounds
    x_min = int(np.floor(min(x_min1, x_min2)))
    x_max = int(np.ceil(max(x_max1, x_max2)))

    # Compute y-coordinates for both radial boundaries at all eta values
    rad1 = rad_dom[0] * np.sin(eta_dom)
    rad2 = rad_dom[-1] * np.sin(eta_dom)
    y_min1 = np.min(-rad1 + center[0])
    y_max1 = np.max(-rad2 + center[0])
    y_max2 = np.max(-rad1 + center[0])
    y_min2 = np.min(-rad2 + center[0])

    # Find overall y-coordinate bounds
    y_min = int(np.floor(min(y_min1, y_min2)))
    y_max = int(np.ceil(max(y_max1, y_max2)))

    return np.array([x_min, x_max + 1]), np.array([y_min, y_max + 1])

def applyTilt(x, y, tilt, detectDist):
    """
    Applies a tilt to Cartesian coordinates.

    Parameters:
    -----------
    x : float
        x-coordinate in pixels.
    y : float
        y-coordinate in pixels.
    tilt : ndarray
        Tilt vector (rotation angles in radians).
    detectDist : float
        Detector distance in millimeters.

    Returns:
    --------
    tuple
        Adjusted (x, y) coordinates after tilt application.
    """
    z = -detectDist  # Set z-coordinate to negative detector distance

    # Create the detector rotation matrix from tilt angles
    R = transforms.xf.makeDetectorRotMat(tilt)

    # Apply rotation matrix to the (x, y, z) vector
    xyz2 = np.array([[x], [y], [z]])
    xyz = np.matmul(R, xyz2)

    return xyz.ravel()[0], xyz.ravel()[1]

def applyTransTilt(x, y, tilt, trans):
    """
    Applies both translation and tilt to Cartesian coordinates.

    Parameters:
    -----------
    x : float
        x-coordinate in pixels.
    y : float
        y-coordinate in pixels.
    tilt : ndarray
        Tilt vector (rotation angles in radians).
    trans : ndarray
        Translation vector (x, y, z) in millimeters.

    Returns:
    --------
    tuple
        Adjusted (x, y) coordinates after applying both translation and tilt.
    """
    z = trans[2]  # Extract z-translation
    x = x - trans[0]  # Subtract x-translation
    y = y - trans[1]  # Subtract y-translation

    # Create the detector rotation matrix from tilt angles
    R = transforms.xf.makeDetectorRotMat(tilt)

    # Apply rotation matrix to the (x, y, z) vector
    xyz2 = np.array([[x], [y], [z]])
    xyz = np.matmul(R, xyz2)

    return xyz.ravel()[0], xyz.ravel()[1]

def omegaToFrame(omega,startFrame=FRAME1,numFrames=NUMFRAMES,omegaRange=OMEG_RANGE,startOmeg = 0):
    '''
    Converts an omega value in radians to its corresponding frame number.

    Parameters:
    -----------
    omega: float
        Omega value in radians, typically in the range (-π, π).
    startFrame: int, optional
        First frame of the scan (default is FRAME1).
    numFrames: int, optional
        Total number of omega frames (default is NUMFRAMES).
    omegaRange: float, optional
        Length of the omega scan in degrees (default is OMEG_RANGE).
    startOmeg: float, optional
        Omega value in radians at `startFrame` (default is 0).

    Returns:
    --------
    frame: int
        Frame number corresponding to the given omega value, ranging 
        from `startFrame` to `startFrame + numFrames`.
    '''
    step = omegaRange/numFrames
    frame = (np.floor((omega*180/np.pi-startOmeg)/step) + startFrame).astype(int)
    return frame
      
def frameToOmega(frame,startFrame=FRAME1,numFrames=NUMFRAMES,omegaRange=OMEG_RANGE,startOmeg = 0):
    '''
    Converts a frame number to its corresponding omega value in radians.
      
    Parameters:
    -----------
    frame: int
        Frame number to be converted.
    startFrame: int, optional
        First frame of the scan (default is FRAME1).
    numFrames: int, optional
        Total number of omega frames (default is NUMFRAMES).
    omegaRange: float, optional
        Length of the omega scan in degrees (default is OMEG_RANGE).
    startOmeg: float, optional
        Omega value in radians at `startFrame` (default is 0).
      
    Returns:
    --------
    omega: float
        Omega value in radians corresponding to the given frame.
    '''
    step = omegaRange/numFrames
    omega = (frame - startFrame)*step + startOmeg
    return omega

def mapOmega(omega):
    '''
    Maps an omega value to the range (-180, 180] degrees.

    Parameters:
    -----------
    omega: float
        Omega value in degrees to be mapped.

    Returns:
    --------
    omega: float
        Omega value mapped to the range (-180, 180].
    '''
    if omega > 180:
        omega = omega - 360
    return omega

def mapDiff(diff):
    '''
    Maps an angular difference to the range (-180, 180] degrees.

    Parameters:
    -----------
    diff: float
        Angular difference in degrees.

    Returns:
    --------
    diff: float
        Angular difference mapped to the range (-180, 180].
    '''
    return (diff + 180) % 360 - 180

def wrapFrame(frm, frm0=FRAME1, numFrms=NUMFRAMES):
    '''
    Wraps a frame number to ensure it stays within the valid frame range.

    Parameters:
    -----------
    frm: int
        Frame number to be wrapped.
    frm0: int, optional
        The starting frame of the scan (default is FRAME1).
    numFrms: int, optional
        Total number of frames in the scan (default is NUMFRAMES).

    Returns:
    --------
    wrappedFrame: int
        Frame number wrapped within the range [frm0, frm0 + numFrms).
    '''
    return np.mod(frm - frm0, numFrms) + frm0

def pathToFile(path):
    '''
    Retrieves and sorts the full file paths of all files in a given directory.

    Parameters:
    -----------
    path: str
        Path to the directory containing the files.

    Returns:
    --------
    fnames: list of str
        A sorted list of full file paths for all files in the specified directory.
    '''
    fnames = os.listdir(path)
    fnames.sort()
    for i in range(len(fnames)):
        fnames[i] = os.path.join(path, fnames[i])
    return fnames

def timeToFile(t, fDir):
    '''
    Constructs file paths for a specific time step by navigating the directory structure.

    Parameters:
    -----------
    t: int
        Time step or identifier for the desired file sequence.
    fDir: str
        Base directory path where time-step-specific directories are located.

    Returns:
    --------
    fnames: list of str
        A sorted list of full file paths for files corresponding to the given time step.
    '''
    dNum = str(t)
    topDir = os.path.join(fDir + dNum, 'ff')
    fnames = pathToFile(topDir)
    return fnames

def getDataFileSequence(dataFile, scanRange):
    '''
    Generates a sequence of data file paths based on a file template and scan range.

    Parameters:
    -----------
    dataFile: str
        Template for the file path, where placeholders can be replaced with values from `scanRange`.
    scanRange: iterable
        A range or list of values to populate the file template with.

    Returns:
    --------
    dataFileSequence: list of str
        A list of full file paths generated from the template and scan range.

    Notes:
    ------
    - The function uses `glob.glob` to find files matching the generated patterns.
    - Assumes each pattern matches exactly one file.
    '''
    dataFileSequence = []
    for scan in scanRange:
        template = dataFile.format(num2=scan)
        pattern = os.path.join(template)
        fname = glob.glob(pattern)[0]
        if len(fname) > 0:
            dataFileSequence.append(fname)
    return dataFileSequence

def findSpots(spotData, grains=None, tths=None, dtth=None, eta=None, deta=None, frm=None):
    '''
    Filters spots from `spotData` based on given conditions such as grain numbers, 
    2theta (tth) values, eta values, and frame indices.

    Parameters:
    -----------
    spotData: dict
        A dictionary containing spot information with the following keys:
        - 'grain_nums': Array of grain numbers.
        - 'etas': Array of eta values.
        - 'tths': Array of two-theta values.
        - 'ome_idxs': Array of frame indices.
    grains: list of int, optional
        List of grain numbers to filter on. Defaults to None (no filtering by grain numbers).
    tths: list of float, optional
        List of two-theta values to filter on. Defaults to None (no filtering by tths).
    dtth: float, optional
        Tolerance for filtering by tths. Required if `tths` is provided.
    eta: float, optional
        Central eta value for filtering. Defaults to None (no filtering by eta).
    deta: float, optional
        Tolerance for filtering by eta. Required if `eta` is provided.
    frm: int, optional
        Frame index to filter on. Defaults to None (no filtering by frame indices).

    Returns:
    --------
    spotInds: ndarray
        Array of indices corresponding to the spots that satisfy the conditions.
    '''
    grain_nums = spotData.get('grain_nums', None)
    etas = spotData.get('etas', None)
    tths_data = spotData.get('tths', None)
    frms = spotData.get('ome_idxs', None)
    
    # Conditions
    cond1 = np.isin(grain_nums, grains) if grains is not None else True
    cond2 = ((etas > eta - deta) & (etas < eta + deta)) if eta is not None else True
    cond3 = (
        np.any([(tths_data > tth - dtth) & (tths_data < tth + dtth) for tth in tths], axis=0) 
        if tths is not None and dtth is not None else True
    )
    cond4 = (frms == frm) if frm is not None else True

    # Combine all conditions
    spotInds = np.where(cond1 & cond2 & cond3 & cond4)[0]

    return spotInds

def collectSpotsData(outPath, spotsPath, sortFlag=False):
    '''
    Collects and processes spot data from multiple `.out` files in a given directory structure, 
    aggregates the data, filters invalid entries, and saves the result as a `.npz` file.

    Parameters:
    -----------
    outPath: str
        Path to the directory where the output `.npz` file will be saved.
    spotsPath: str
        Path to the directory containing the subdirectories with `.out` files.
    sortFlag: bool, optional
        If True, sorts the data by omega values before saving. Default is False.

    Process:
    --------
    - Reads `.out` files within subdirectories of `spotsPath`.
    - Extracts relevant data fields from each file (e.g., coordinates, grain numbers, angles).
    - Appends data across files and filters invalid entries (e.g., missing or NaN values).
    - Optionally sorts the data by omega values (`omes`) if `sortFlag` is set.
    - Saves the processed data to a compressed `.npz` file.

    Output:
    -------
    Saves a `.npz` file in `outPath` containing:
    - `Xs`, `Ys`: Predicted X and Y coordinates.
    - `Xm`, `Ym`: Measured X and Y coordinates.
    - `id_nums`: IDs of the spots.
    - `tths`, `etas`, `omes`: Measured two-theta, eta, and omega values.
    - `tths_pred`, `etas_pred`, `omes_pred`: Predicted two-theta, eta, and omega values.
    - `ome_idxs`: Frame indices corresponding to omega values.
    - `grain_nums`: Grain numbers for each spot.
    - `PID`: Phase IDs.
    - `H`, `K`, `L`: Miller indices.

    Notes:
    ------
    - Files with invalid or missing data are automatically filtered out.
    - Assumes `.out` files have specific columns such as 'pred X', 'meas X', 'meas tth', etc.
    '''
    all_entries = os.listdir(spotsPath)
    directories = [entry for entry in all_entries if os.path.isdir(os.path.join(spotsPath, entry))]
    fold_names = sorted(directories)

    created = False

    for fold_name in fold_names:
        file_names = sorted(os.listdir(os.path.join(spotsPath, fold_name)))
        print(fold_name)

        for file_name in file_names:
            if file_name.endswith(".out"):
                file_path = os.path.join(spotsPath, fold_name, file_name)

                # Load .out file
                df = pd.read_csv(file_path, sep='\\s+', engine='python')

                # Extract data
                grain_number = int(file_name[-7:-4])
                new_data = {
                    'Xs': df['pred X'].to_numpy(),
                    'Ys': df['pred Y'].to_numpy(),
                    'Xm': df['meas X'].to_numpy(),
                    'Ym': df['meas Y'].to_numpy(),
                    'id_nums': df['# ID'].to_numpy(),
                    'PID': df['PID'].to_numpy(),
                    'H': df['H'].to_numpy(),
                    'K': df['K'].to_numpy(),
                    'L': df['L'].to_numpy(),
                    'grain_nums': grain_number * np.ones(df.shape[0]),
                    'tths': df['meas tth'].to_numpy(),
                    'etas': df['meas eta'].to_numpy(),
                    'omes': df['meas ome'].to_numpy(),
                    'tths_pred': df['pred tth'].to_numpy(),
                    'etas_pred': df['pred eta'].to_numpy(),
                    'omes_pred': df['pred ome'].to_numpy(),
                }

                # Initialize or append
                if not created:
                    combined_data = new_data
                    created = True
                else:
                    for key in combined_data:
                        combined_data[key] = np.append(combined_data[key], new_data[key])

    # Filter invalid data
    invalid_mask = (
        (combined_data['id_nums'] == -999) | 
        np.isnan(combined_data['tths']) | 
        np.isnan(combined_data['etas'])
    )
    for key in combined_data:
        combined_data[key] = np.delete(combined_data[key], invalid_mask)

    # Convert omegas to frame indices
    combined_data['ome_idxs'] = omegaToFrame(combined_data['omes'])

    # Sort data if required
    if sortFlag:
        sort_indices = np.argsort(combined_data['omes'])
        for key in combined_data:
            combined_data[key] = combined_data[key][sort_indices]

    # Save to .npz file
    save_file = os.path.join(outPath, 'spots.npz')
    np.savez(save_file, **combined_data)

def getSpotID(spotData, k):
    """
    Retrieves the spot ID for a specific index `k` from the spot data.

    Parameters:
    -----------
    spotData: dict
        Dictionary containing spot data with keys like 'grain_nums', 'PID', etc.
    k: int
        Index of the spot.

    Returns:
    --------
    list
        List containing [grain number, PID, H, K, L, ID].
    """
    keys = ['grain_nums', 'PID', 'H', 'K', 'L', 'id_nums']
    spot_id = [spotData[key][k] for key in keys]
    return spot_id

def matchSpotID(spotData, spot_id, k):
    """
    Finds the index of a spot in `spotData` that matches the given `spot_id`.

    Parameters:
    -----------
    spotData: dict
        Dictionary containing spot data with keys like 'grain_nums', 'PID', etc.
    spot_id: list
        List containing [grain number, PID, H, K, L, ID] of the spot to match.
    k: int
        Original index of the spot for reference in case of no match.

    Returns:
    --------
    int or float
        Index of the matching spot, or NaN if no match is found.
    """
    # Extract spot data fields
    fields = ['grain_nums', 'PID', 'H', 'K', 'L', 'id_nums']
    grNum, PID, H, K, L, idNum = [spotData[field] for field in fields]

    # Match based on grain number, PID, and Miller indices
    bin_array = (grNum == spot_id[0]) & (PID == spot_id[1]) & \
                (H == spot_id[2]) & (K == spot_id[3]) & (L == spot_id[4])
    match_ind = np.where(bin_array)[0]

    # Refine match if multiple indices found
    if len(match_ind) > 1:
        diffs = abs(idNum[match_ind] - spot_id[5])
        valid_matches = diffs < 50
        match_ind = match_ind[valid_matches]
        diffs = diffs[valid_matches]

        if len(match_ind) > 1:
            match_ind = match_ind[np.argmin(diffs)]

    # Handle no matches
    if not match_ind.size:
        print(f'Match not found for spot {k}')
        return np.nan

    return match_ind.item()  # Return scalar value

def estMEANomega(track):
    """
    Estimates the mean omega value for a given track.

    Parameters:
    -----------
    track: list of dicts
        Each dictionary contains keys 'roi' (region of interest array) 
        and 'frm' (frame index).

    Returns:
    --------
    float
        Estimated mean omega value.
    """
    step = OMEG_RANGE / NUMFRAMES

    if len(track) == 1:
        return frameToOmega(track[0]['frm'])

    roiOmega = np.array([np.nansum(t['roi']) for t in track])
    omegaRange = np.array([frameToOmega(t['frm']) for t in track])
    weighted_mean_index = np.sum(np.arange(len(track)) * roiOmega) / np.sum(roiOmega)

    ind1 = int(np.floor(weighted_mean_index))
    ind2 = int(np.ceil(weighted_mean_index))

    meanOmega = (
        omegaRange[ind1] + (weighted_mean_index - ind1) * step
        if omegaRange[ind1] > omegaRange[ind2]
        else weighted_mean_index * step
    )

    return mapOmega(meanOmega)  # Map to -180° to 0° if necessary

def estFWHMomega(track):
    """
    Estimates the Full-Width Half-Maximum (FWHM) of omega for a given track.

    Parameters:
    -----------
    track: list of dicts
        Each dictionary contains keys 'roi' (region of interest array) 
        and 'frm' (frame index).

    Returns:
    --------
    float
        Estimated FWHM of omega.
    """
    step = OMEG_RANGE / NUMFRAMES

    if len(track) == 1:
        return 0

    roiOmega = np.array([np.nansum(t['roi']) for t in track])
    weighted_mean_index = np.sum(np.arange(len(track)) * roiOmega) / np.sum(roiOmega)

    # Compute variance
    varOmega = np.sum(
        roiOmega * ((np.arange(len(track)) - weighted_mean_index) ** 2)
    ) / np.sum(roiOmega) * step**2

    # Calculate FWHM
    fwhmOmega = 2 * np.sqrt(2 * np.log(2) * varOmega)

    return fwhmOmega

def xyToEtaTth(x, y, params):
    """
    Converts Cartesian coordinates (x, y) to eta and 2-theta.

    Parameters:
    -----------
    x, y: float or np.ndarray
        Cartesian coordinates.
    params: dict
        Dictionary containing detector parameters.

    Returns:
    --------
    tuple
        eta (float or np.ndarray): Azimuthal angle in radians.
        tth (float or np.ndarray): 2-theta angle in radians.
    """
    detectDist, _, _, _ = loadYamlData(params)
    eta = np.arctan2(y, x)
    rad = np.sqrt(x**2 + y**2)
    tth = np.arctan(rad / detectDist)
    return eta, tth

def xyToEtaTthRecenter(x, y, params):
    """
    Converts Cartesian coordinates (x, y) to eta and 2-theta after recentering.

    Parameters:
    -----------
    x, y: float or np.ndarray
        Cartesian coordinates.
    params: dict
        Dictionary containing detector parameters, including translation.

    Returns:
    --------
    tuple
        eta (float or np.ndarray): Azimuthal angle in radians.
        tth (float or np.ndarray): 2-theta angle in radians.
    """
    detectDist, _, trans, _ = loadYamlData(params)
    x += trans[0]
    y += trans[1]
    eta = np.arctan2(y, x)
    rad = np.sqrt(x**2 + y**2)
    tth = np.arctan(rad / detectDist)
    return eta, tth
    
def etaTthToPix(eta, tth, etaRoi, tthRoi, params):
    """
    Converts eta and 2-theta to pixel coordinates in the detector.

    Parameters:
    -----------
    eta, tth: float or np.ndarray
        Eta (azimuthal) and 2-theta angles in radians.
    etaRoi, tthRoi: tuple
        Region of interest for eta and 2-theta, respectively.
    params: dict
        Dictionary containing detector parameters, including ROI size.

    Returns:
    --------
    tuple
        row_pos, col_pos: Pixel coordinates.
    """
    roiSize = params['roiSize']
    detectDist, mmPerPixel, _, _ = loadYamlData(params, tthRoi, etaRoi)
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tthRoi, etaRoi, roiSize)

    # Calculate radial and azimuthal positions
    rad = np.tan(tth) * detectDist / mmPerPixel
    row_pos = (rad - rad_dom[0]) / (rad_dom[-1] - rad_dom[0]) * (roiSize[0] - 1)
    col_pos = (eta - eta_dom[0]) / (eta_dom[-1] - eta_dom[0]) * (roiSize[1] - 1)

    return row_pos, col_pos
    
def pixToEtaTth(p1, p2, tthRoi, etaRoi, params):
    """
    Converts pixel coordinates to eta and 2-theta.

    Parameters:
    -----------
    p1, p2: float
        Pixel coordinates.
    tthRoi, etaRoi: tuple
        Region of interest for 2-theta and eta, respectively.
    params: dict
        Dictionary containing detector parameters, including ROI size.

    Returns:
    --------
    tuple
        etaNew, tthNew: New eta and 2-theta values.
        deta, dtth: Step sizes in eta and 2-theta.
    """
    roiSize = params['roiSize']
    detectDist, mmPerPixel, _, _ = loadYamlData(params, tthRoi, etaRoi)
    rad_dom, eta_dom = polarDomain(detectDist, mmPerPixel, tthRoi, etaRoi, roiSize)

    # Calculate step sizes
    deta = abs(eta_dom[1] - eta_dom[0])
    hypot = detectDist * np.cos(tthRoi)
    dtth = np.arctan(mmPerPixel / hypot)

    # Determine new eta and radial positions
    i1, j1 = int(np.floor(p1)), int(np.floor(p2))
    etaNew = eta_dom[i1] + deta * (p1 % 1)
    radNew = rad_dom[j1] + (p2 % 1)
    tthNew = np.arctan(radNew * mmPerPixel / detectDist)

    return etaNew, tthNew, deta, dtth

def indToFrame(ind, frmRoi, numFrms):
    """
    Maps an index to a frame number within a range of frames.

    Parameters:
    -----------
    ind : int
        The index of the frame.
    frmRoi : int
        The starting frame number.
    numFrms : int
        The total number of frames in the range.

    Returns:
    --------
    outputFrame : int
        The calculated frame number.
    """
    # Ensure the index is within bounds
    if ind < 0 or ind >= numFrms:
        raise ValueError(f"Index {ind} is out of bounds for numFrms {numFrms}")
    
    # Calculate the offset from the middle frame
    offset = ind - (numFrms // 2)
    
    # Return the frame number
    return frmRoi + offset
