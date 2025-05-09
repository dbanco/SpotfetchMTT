# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:14:23 2025

@author: dpqb1
"""
import redis
import json
import psycopg2
import time
import pickle
import numpy as np
import os

import utilities as util
from mtt_system import MTTSystem
from mtt_framework.detection import ThresholdingDetector
from mtt_framework.feature_extraction import BasicFeatureExtractor
from mtt_framework.mht_tracker import MHTTracker
from mtt_framework.state_model import KalmanModel

# Connect to Redis
redis_client = redis.Redis(host="lnx7108.classe.cornell.edu", port=6379, decode_responses=True)

# PostgreSQL connection details
DB_HOST = "lnx7108.classe.cornell.edu"
DB_NAME = "postgres"
DB_USER = "dbanco"
DB_PASS = "yourpassword"

# File paths
OMEGA_FRAME_DIR = "/omega_frames"
TRACKER_SAVE_DIR = "/tracker_states"

# Ensure tracker save directory exists
os.makedirs(TRACKER_SAVE_DIR, exist_ok=True)

def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Works for both arrays and scalars wrapped in arrays
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Converts np.int64, np.float64 → Python int/float
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    else:
        return obj  # Leave as-is

def write_to_database(region_id, track_id, scan_number, state, overlapping, detected=True):
    """Writes tracking results to PostgreSQL, storing all features as JSON."""
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        conn.autocommit = True
        cursor = conn.cursor()
        
        query = """
        INSERT INTO regions (region_id)
        VALUES (%s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(query, (region_id,))
        conn.commit()
        
        query = """
        INSERT INTO tracks (region_id, track_id, first_detected_scan)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(query, (region_id, track_id, scan_number))
        conn.commit()

        # Format the feature dictionary from `state`
        features = {
            "com": state["com"],
            "velocity": state["velocity"],
            "bbox": state["bbox"],
            "bbox_center": state["bbox_center"],
            "bbox_size": state["bbox_size"],
            "intensity": state["intensity"],
            "principal_axes": state["principal_axes"],
            "variance": state["variance"],
        }
        features = convert_for_json(features)
        
        # print("Feature types before serialization:")
        # for key, value in features.items():
        #     print(f"{key}: type = {type(value)}", flush=True)

        # SQL query to insert/update measurement
        query = """
        INSERT INTO measurements (region_id, track_id, scan_number, detected, overlapping, features)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (region_id, track_id, scan_number) DO UPDATE
        SET detected = EXCLUDED.detected, features = EXCLUDED.features;
        """

        # Execute query
        cursor.execute(query, (region_id, track_id, scan_number, 
                               detected, overlapping, json.dumps(features)))
        conn.commit()
        
        cursor.close()
        conn.close()
        print(f"Wrote tracking data for Track {track_id} (Region {region_id}, Scan {scan_number}).",flush=True)

    except Exception as e:
        print(f"Database write error: {e}")

def load_mtt_system(scan_number, region_id):
    """Loads or initializes an MTT system for the given scan number."""
    tracker_file = os.path.join(TRACKER_SAVE_DIR, f"tracker_region_{region_id}.pkl")

    if scan_number == 0 or not os.path.exists(tracker_file):
        print(f"Initializing new tracker for Region {region_id}.",flush=True)
        # Initialize detector, feature extractor, and tracker
        spot_detector = ThresholdingDetector(threshold=20)
        track_model = KalmanModel(
            initial_state={'com': np.zeros(3), 'velocity': np.zeros(3), 'acceleration': np.zeros(3), 'bbox': np.zeros(6)},
            feature_extractor=BasicFeatureExtractor(),
            process_noise=1e-5, measurement_noise=1e-5, dt=1
        )
        mht_tracker = MHTTracker(track_model=track_model, n_scan_pruning=4, plot_tree=False)
        mtt_system = MTTSystem(spot_detector=spot_detector, track_model=track_model, tracker=mht_tracker)
    else:
        print(f"Loading existing tracker from {tracker_file}.",flush=True)
        with open(tracker_file, "rb") as f:
            mtt_system = pickle.load(f)

    return mtt_system

def save_mtt_system(mtt_system, region_id):
    """Saves the current state of the MTT system."""
    tracker_file = os.path.join(TRACKER_SAVE_DIR, f"tracker_region_{region_id}.pkl")
    with open(tracker_file, "wb") as f:
        pickle.dump(mtt_system, f)
    print(f"💾 Saved tracker state for Region {region_id}.")

def process_region(region_id, scan_number, frames):
    """Loads frame data, processes it with MTT system, writes results, and saves state."""
    data = []
    for frame_number in frames:
        frame_path = os.path.join(OMEGA_FRAME_DIR, f"region_{region_id}_frame_{frame_number}_scan_{scan_number}.npy")
        if not os.path.exists(frame_path):
            print(f"🚨 Missing file: {frame_path}, skipping...",flush=True)
            return
        data.append(np.load(frame_path))
        
    region = np.stack(data)
    print(f"Loaded frames at (Scan {scan_number}) for Region {region_id}.",flush=True)

    # Load or initialize tracker
    mtt_system = load_mtt_system(scan_number, region_id)

    # Process the frame
    success = mtt_system.process_frame(region, scan_number)
    print(f"Processed frame {frame_number} for Region {region_id}.",flush=True)

    # Write results to PostgreSQL
    if success:
        for node in mtt_system.tracker.tree.leaf_nodes:
            if node.best and node.scan == scan_number:
                if len(node.track_id) > 1:
                    overlapping = True
                else:
                    overlapping = False
                for t_id in node.track_id:
                    write_to_database(region_id, t_id, scan_number, node.track.state, overlapping)
        
        # Save updated tracker state
        save_mtt_system(mtt_system, region_id)

def fetch_and_process_jobs():
    """Continuously fetches tracking jobs from Redis and processes them."""
    print("Tracker is waiting for jobs...",flush=True)

    while True:
        job_data = redis_client.lpop("tracking_jobs")

        if job_data:
            job = json.loads(job_data)
            region_id = job["region_id"]
            scan_number = job["scan_number"]
            required_frames = set(util.wrapFrame(np.arange(job["start_frame"], job["end_frame"] + 1)))

            # Check if all required frames exist
            available_frames = redis_client.smembers(f"region_{region_id}_scan_{scan_number}_frames")
            available_frames = {int(f) for f in available_frames}  # Convert to integers

            missing_frames = required_frames - available_frames

            if missing_frames:
                print(f"Waiting for missing frames: {missing_frames} for Region {region_id}, Scan {scan_number}...")
                time.sleep(0.5)
                redis_client.rpush("tracking_jobs", json.dumps(job))  # Requeue job
                continue

            print(f"Processing {job}",flush=True)
            process_region(region_id=region_id, scan_number=scan_number, frames=required_frames)

        else:
            print("🔍 No jobs available, waiting...")
            time.sleep(3)  # Prevent excessive CPU usage


def initialize_database():
    """Ensures the database schema exists with composite keys for region-based tracking."""
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute("DROP SEQUENCE IF EXISTS regions_region_id_seq CASCADE;")
        
        # Create `regions` table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS regions (
            region_id SERIAL PRIMARY KEY,
            description TEXT
        );
        """)

        # Create `tracks` table (with composite primary key)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            region_id INT NOT NULL REFERENCES regions(region_id) ON DELETE CASCADE,
            track_id INT NOT NULL,
            first_detected_scan INT NOT NULL,
            last_detected_scan INT,
            PRIMARY KEY (region_id, track_id)  -- Composite Primary Key
        );
        """)

        # Create `measurements` table (linked to `tracks`)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id SERIAL PRIMARY KEY,
            region_id INT NOT NULL,
            track_id INT NOT NULL,
            scan_number INT NOT NULL,
            detected BOOLEAN NOT NULL,
            overlapping BOOLEAN NOT NULL,
            features JSONB,
            UNIQUE (region_id, track_id, scan_number),
            FOREIGN KEY (region_id, track_id) REFERENCES tracks(region_id, track_id) ON DELETE CASCADE
        );
        """)

        # Create indexes for fast queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_scan ON measurements(scan_number);
        CREATE INDEX IF NOT EXISTS idx_measurements_track ON measurements(region_id, track_id);
        """)
        
        # Index for filtering detected spots efficiently
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_detected_true 
        ON measurements(scan_number) WHERE detected = TRUE;
        """)
        
        # Index for filtering detected & non-overlapping spots efficiently
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_detected_nonoverlapping 
        ON measurements(scan_number) 
        WHERE detected = TRUE AND (features->>'overlapping')::BOOLEAN = FALSE;
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized with composite keys for region-based tracking.")

    except Exception as e:
        print(f"Database initialization error: {e}")

if __name__ == "__main__":
    initialize_database()
    fetch_and_process_jobs()



