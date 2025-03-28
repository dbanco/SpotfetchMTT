# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:33:36 2025

@author: dpqb1
"""
import sys
import redis
import json
import os
import re
import time
from watchdog.observers.polling import PollingObserver  # Ensure polling mode is used
from watchdog.events import FileSystemEventHandler
import importlib.util

REDIS_HOST = "lnx7108.classe.cornell.edu"
REDIS_PORT = 6379
redis_client = redis.Redis(host="lnx7108.classe.cornell.edu", port=6379, decode_responses=True)

REGION_DIR = "/region_files"
REGION_PATTERN = re.compile(r"region_(\d+)_scan_(\d+)\.npy")

print(f"Region Listener started. Watching: {REGION_DIR}", flush=True)

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

class RegionHandler(FileSystemEventHandler):
    """Detects new omega frames and adds them to Redis."""
    def __init__(self, jobs, redis_client):
        self.jobs = jobs
        self.redis = redis_client
        
    def on_created(self, event):
        print(f"File event detected: {event.src_path}", flush=True)
        
        if event.is_directory:
            print(f"Ignoring directory: {event.src_path}", flush=True)
            return
        
        filename = os.path.basename(event.src_path)
        match = REGION_PATTERN.match(filename)

        if match:
            region_id = int(match.group(1))
            scan_number = int(match.group(2))
            print(f"New Region detected: Region {region_id}, Scan {scan_number}", flush=True)
            for job in self.jobs[:]:  
                if job["region_id"] == region_id and scan_number == job["scan_number"]:
                    self.redis.rpush("tracking_jobs", json.dumps(job))
                    print('Pushed to queue')
        else:
            print(f"Skipping unrecognized file: {filename}", flush=True)

def watch_for_regions(jobs,redis_client):
    print("Starting watchdog observer...", flush=True)
    observer = PollingObserver()  # Use PollingObserver instead of Observer()
    event_handler = RegionHandler(jobs,redis_client)
    observer.schedule(event_handler, REGION_DIR, recursive=False)
    observer.start()

    try:
        while True:
            print("Checking for new files...", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping watchdog observer...", flush=True)
        observer.stop()
    observer.join()

if __name__ == "__main__":
    jobs = load_jobs("job_config.json")
    watch_for_regions(jobs,redis_client)




