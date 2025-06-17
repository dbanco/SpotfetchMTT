# -*- coding: utf-8 -*-
"""
mtt_framework

Description:
Module for tracking x-ray diffraction spots in 3D using mulitple hypotheses test
multitarget tracking
Created on Tue Feb 18 09:59:02 2025

@author: dpqb1
"""

class MTTSystem:
    """
    Multi-Target Tracking System that integrates modular components for:
    - Spot detection
    - State modeling
    - Tracking
    """

    def __init__(self, spot_detector, track_model, tracker):
        """
        Initializes the MTT system with user-defined components.

        Parameters:
        - spot_detector: An instance of a spot detection module.
        - feature_extractor: An instance of a feature extractor.
        - track_model: An instance of a state model for tracking.
        - tracker: An instance of a tracking algorithm (e.g., MHT).
        """
        self.spot_detector = spot_detector
        self.track_model = track_model
        self.tracker = tracker
    
    def process_frame(self, frame, scan):
        """
        Processes a single frame of data through the MTT pipeline.

        Parameters:
        - frame: The 3D input data frame containing detection information.

        Returns:
        - tracks: The updated set of tracked objects.
        """

        assert len(frame.shape) == 3, "Input data must be 3 dimensional"
        
        # Step 1: Detection
        blobs, num_blobs = self.spot_detector.detect(frame)
        
        # Step 2: Reduce pixel detections to measurements
        measurements = self.track_model.get_measurements(frame,blobs)
        
        if len(measurements) > 0:
            # Step 3: Run tracker
            self.tracker.process_measurements(measurements, scan)
            return True
        else:
            print("No measurements to track")
            return False  


