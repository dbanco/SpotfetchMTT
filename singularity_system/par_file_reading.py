#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:24:15 2025

@author: dbanco
"""

import os
import pandas as pd
import json

# data_dir = r"/nfs/chess/raw/2024-2/id3a/miller-3528-c/c103-1-ff-1"
# par_file = os.path.join(data_dir,"id3a-rams2_ff_scan_layers-dexela-c103-1-ff-1.par")
# json_file = os.path.join(data_dir,"id3a-rams2_ff_scan_layers-dexela-c103-1-ff-1.json")

data_dir = r"/nfs/chess/raw/2025-1/id3a/shanks-3731-d/ti-2-test/"
par_file = os.path.join(data_dir,"rams2-slew_ome.par")
json_file = os.path.join(data_dir,"rams2-slew_ome.json")

# Load column name mapping from JSON
with open(json_file, "r") as f:
    col_map = json.load(f)

# Convert JSON keys (strings) to integers and sort
col_names = [col_map[str(i)] for i in sorted(map(int, col_map.keys()))]

# Load the data using whitespace as delimiter
df = pd.read_csv(par_file,sep=r'\s+', header=None, names=col_names)

# Show the parsed dataframe
for scan_n in df['SCAN_N']:
    print([scan_n])
    
    
    