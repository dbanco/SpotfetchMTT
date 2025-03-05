# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:06:38 2025

@author: Bahar
"""

class Detection:
    def __init__(self, blob_label, mask, x_masked):
        self.blob_label = blob_label
        self.mask = mask
        self.x_masked = x_masked