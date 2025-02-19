# -*- coding: utf-8 -*-
"""

full_mtt_system_test

Created on Tue Feb 18 17:22:31 2025

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mtt_framework.detection import HDoGDetector
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.state_model import ConstantVelocityStateModel
from mtt_framework.mht_tracker import MHTTracker
from mtt_framework.mtt_system import MTTSystem

from data.synthetic_spot_data import generate_synthetic_data

def test_basic_mtt():
    """
    Runs a basic test for the MTT system.
    """
    print("Running Basic MHT MTT System Test...")

    # Initialize components
    detector = HDoGDetector()
    feature_extractor = BasicFeatureExtractor()
    state_model = ConstantVelocityStateModel(feature_extractor)
    tracker = MHTTracker()

    # Initialize the MTT System
    mtt_system = MTTSystem(detector, feature_extractor, state_model, tracker)

    # Generate test data
    test_data = generate_synthetic_data()

    # Run tracking
    results = mtt_system.run_tracking(test_data)

    # Print results
    print("Tracking Results:", results)

if __name__ == "__main__":
    test_basic_mtt()
