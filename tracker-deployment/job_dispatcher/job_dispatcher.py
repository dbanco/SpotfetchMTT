# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:21:12 2025

@author: dpqb1
"""
import redis
import json
import time
import os
import numpy as np
import utilities as util

# Connect to Redis
redis_client = redis.Redis(host="redis_queue", port=6379, decode_responses=True)

# Load spot data
topPath = "/data"
spotsFile = os.path.join(topPath,"spots.npz")
spotData = np.load(spotsFile)

# Define region size
params = {}
params['detector'] = 'eiger'
params['imSize'] = (5000,5000)
params['yamlFile'] = os.path.join(topPath,"eiger16M_monolith_mruby_062224_FINAL.yml")
params['roiSize'] = [35,35,11]
dfrm = (params['roiSize'][2]-1)/2

# Initialize jobs for tracking jobs for scan 0
print("Setting up initial set of jobs to scan for",flush=True)
TRACKING_JOBS = []
for spotInd in range(5):
    x = spotData['Xm'][spotInd]
    y = spotData['Ym'][spotInd]
    frm = int(spotData['ome_idxs'][spotInd])
    eta, tth = util.xyToEtaTthRecenter(x,y,params)

    TRACKING_JOBS.append({
        "region_id": spotInd, 
        "start_frame": int(util.wrapFrame(frm-dfrm)), 
        "end_frame": int(util.wrapFrame(frm+dfrm)), 
        "scan_number": 0})

def check_frames_available(region_id, scan_number, start_frame, end_frame):
    """
    Check if all frames in the range [start_frame, end_frame] exist in Redis.
    Redis Key: "region_{region_id}_scan_{scan_number}_frames"
    """
    redis_key = f"region_{region_id}_scan_{scan_number}_frames"  # Corrected key format
    available_frames = redis_client.smembers(redis_key)
    needed_frames = {str(int(frame)) for frame in util.wrapFrame(np.arange(start_frame, end_frame + 1))}
    
    return needed_frames.issubset(available_frames)

def watch_for_jobs():
    """Continuously checks Redis and adds tracking jobs when frames are ready."""
    print("Job Dispatcher is watching for available frames...",flush=True)
    
    global TRACKING_JOBS
    print(TRACKING_JOBS,flush=True)

    while True:
        new_jobs = []  # Store next scan jobs

        for job in TRACKING_JOBS[:]:  # Iterate over a copy
            region_id = job["region_id"]
            scan_number = job["scan_number"]
            start_frame = int(job["start_frame"])
            end_frame = int(job["end_frame"])

            if check_frames_available(region_id, scan_number, start_frame, end_frame):
                # Submit job to Redis queue
                redis_client.rpush("tracking_jobs", json.dumps(job))
                print(f"Added tracking job: {job}",flush=True)

                # Automatically queue the next scan for the same region
                new_jobs.append({
                    "region_id": region_id,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "scan_number": scan_number + 1  # Move to next scan
                })

                # Remove the completed job
                TRACKING_JOBS.remove(job)

        # Add new scan jobs
        TRACKING_JOBS.extend(new_jobs)

        time.sleep(0.5)  # Check every 0.5 seconds

if __name__ == "__main__":
    watch_for_jobs()
