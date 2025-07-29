#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:47:43 2025

@author: dbanco
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory for utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utilities as util

def draw_region(ax, tth, eta_range, detector_distance, mm_per_pixel, im_size, color='r'):
    eta_vals = np.linspace(eta_range[0], eta_range[1], 100)
    radius = np.tan(tth) * detector_distance / mm_per_pixel
    x = radius * np.cos(eta_vals) + im_size[1] / 2
    y = radius * np.sin(eta_vals) + im_size[0] / 2
    ax.plot(x, y, color=color, linewidth=1)

def visualize_regions(img_path_list, yaml_file, config_file, frame_index, params):
    # Load an image slice
    img = util.loadDexImg(img_path_list, params, frame_index)

    # Load tracking region config
    with open(config_file, "r") as f:
        jobs = json.load(f)

    # Geometry info
    yaml_data = util.read_yaml(yaml_file)
    mm_per_pixel = yaml_data['detectors']['ff2']['pixels']['size'][0]
    detector_distance = yaml_data['detectors']['ff1']['transform']['translation'][2]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, vmax=500, cmap="gray")
    ax.set_title(f"Tracking Regions (Frame {frame_index})")
    ax.axis("off")

    for job in jobs:
        draw_region(ax, job["tth"], job["eta"], detector_distance, mm_per_pixel, params['imSize'], color='lime')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    topPath = "/nfs/chess/user/dbanco/c103_processing"
    dataDir = r"/nfs/chess/raw/2024-2/id3a/miller-3528-c/c103-1-ff-1"

    params = {
        'detector': 'dexela',
        'imSize': (4000, 6400),
        'yamlFile': os.path.join(topPath, "dexelas_calibrated_ruby_0504_v01.yml"),
        'roiSize': [30, 30, 11],
    }

    # Choose some omega frame
    omega_frame_index = 100
    
    # Provide two-panel image filenames
    ff_num = 96
    dataFile1 = os.path.join(dataDir, "2", "ff", f"ff1_{ff_num:06d}.h5")
    dataFile2 = os.path.join(dataDir, "2", "ff", f"ff2_{ff_num:06d}.h5")
    image_paths = [dataFile1, dataFile2]

    config_file = "tracking_jobs_config.json"

    visualize_regions(
        img_path_list=image_paths,
        yaml_file=params['yamlFile'],
        config_file=config_file,
        frame_index=omega_frame_index,
        params=params
    )