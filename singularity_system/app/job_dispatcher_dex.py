# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:21:12 2025

@author: dpqb1
"""
import redis
import json
import time
import os
import importlib.util

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
JOB_INPUT_FILE = os.getenv("JOB_INPUT", "job_config.json")  # Default JSON file

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# --- Load jobs from file or module ---
def load_jobs(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            jobs = json.load(f)
    elif path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("job_config", path)
        job_def = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_def)
        jobs = job_def.generate_jobs()
    else:
        raise ValueError("Unsupported job input format: must be .json or .py")
    return jobs

# --- Check if all required frames are available in Redis ---
def check_frames_available(region_id, scan_number, start_frame, end_frame):
    """
    Check if all frames in the range [start_frame, end_frame] exist in Redis.
    Redis Key: "region_{region_id}_scan_{scan_number}_frames"
    """
    redis_key = f"region_{region_id}_scan_{scan_number}_frames"
    available_frames = redis_client.smembers(redis_key)
    needed_frames = {str(f) for f in range(start_frame, end_frame + 1)}
    return needed_frames.issubset(available_frames)

# --- Main dispatcher loop ---
def dispatch_jobs(jobs):
    """Continuously checks Redis and adds tracking jobs when frames are ready."""
    print("Job Dispatcher is watching for available frames...", flush=True)
    TRACKING_JOBS = jobs

    while True:
        new_jobs = []

        for job in TRACKING_JOBS[:]:  # Copy
            region_id = job["region_id"]
            scan_number = job["scan_number"]
            start_frame = int(job["start_frame"])
            end_frame = int(job["end_frame"])
            eta_inds = job["eta_inds"]
            eta = job["eta"]
            tth = job["tth"]

            if check_frames_available(region_id, scan_number, start_frame, end_frame):
                redis_client.rpush("tracking_jobs", json.dumps(job))
                print(f"Added tracking job: {job}", flush=True)

                # Add job for next scan                
                jobs.append({
                    "region_id": region_id,
                    "scan_number": scan_number + 1,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "eta_inds": eta_inds,
                    "eta": eta,
                    "tth": tth
                })

                TRACKING_JOBS.remove(job)

        TRACKING_JOBS.extend(new_jobs)
        time.sleep(0.5)

# --- Entry point ---
if __name__ == "__main__":
    try:
        jobs = load_jobs(JOB_INPUT_FILE)
        dispatch_jobs(jobs)
    except Exception as e:
        print(f"Job Dispatcher initialization failed: {e}", flush=True)
