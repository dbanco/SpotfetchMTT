# Multi-Target Tracking (MTT) Framework

## Overview
The **MTT Framework** is a modular system for multi-target tracking (MTT). It enables flexible integration of different:
- **Spot detection** methods
- **Feature extraction** techniques
- **State models** for tracking
- **Multiple Hypothesis Tracking (MHT)** implementations

This design allows users to swap out components without modifying the rest of the system.

## Repository Structure
```plaintext
mtt_project/
│
├── mtt_system.py        # Main script that integrates detection, tracking, and state modeling
│
├── mtt_framework/       # Core tracking framework
│   ├── __init__.py      # Package initialization
│   ├── detection.py     # Spot detection methods
│   ├── feature_extraction.py # Feature extraction methods
│   ├── state_model.py   # State models for tracking
│   ├── mht_tracker.py   # Multiple Hypothesis Tracker implementation
│
├── experiments/         # Scripts for testing different configurations
│   ├── test_mtt.py      # Example script for running the system
│   ├── benchmark.py     # Performance evaluation
│
├── data/                # Stores datasets (raw and processed)
│
├── results/             # Stores logs, plots, and outputs
│   ├── logs/            # Tracking logs
│   ├── plots/           # Visualization outputs
│   ├── models/          # Saved models or checkpoints
│
├── README.md            # Overview and setup guide
├── requirements.txt     # List of dependencies
├── setup.py             # (Optional) Installable package setup
