# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:35:27 2025

Hypothesis generation test

@author: dpqb1
"""


import numpy as np
from itertools import combinations, chain

def all_combinations_excluding_singletons(iterable):
    """Generate all combinations except singletons."""
    lst = list(iterable)
    return chain.from_iterable(combinations(lst, r) for r in range(2, len(lst) + 1))


tracks = [[],[],[],[],[]]
measurements = [[],[],[],[],[],[]]

m2ta = np.array([[1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])

n_measurements, n_tracks = m2ta.shape

#### Existing track associations ####
for k, track in enumerate(tracks):
    # 1. Deaths
    track.append('death')
    associated_ms = np.where(m2ta[:,k])[0]
    
    # 2. Single associations
    for m in associated_ms:
        track.append(m)
                
    # 4. Consider the splitting of a track that was previously overlapping

# 5. Births of unassociated measurements
unassociated_ms = np.where(np.sum(m2ta,1) == 0)[0]
for m in unassociated_ms:
    tracks.append(['birth'])


#### Existing track associations ####
for m, measurement in enumerate(measurements):
    measurement.append('birth')
    associated_ks = np.where(m2ta[m])[0]
    
    for k in associated_ks:
        measurement.append(k)
        
    