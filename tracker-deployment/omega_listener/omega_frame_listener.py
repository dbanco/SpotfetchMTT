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

redis_client = redis.Redis(host="redis_queue", port=6379, decode_responses=True)
OMEGA_FRAME_DIR = "/omega_frames"
OMEGA_FRAME_PATTERN = re.compile(r"region_(\d+)_frame_(\d+)_scan_(\d+)\.npy")

print(f"Omega Frame Listener started. Watching: {OMEGA_FRAME_DIR}", flush=True)

class OmegaFrameHandler(FileSystemEventHandler):
    """Detects new omega frames and adds them to Redis."""

    def on_created(self, event):
        print(f"File event detected: {event.src_path}", flush=True)
        
        if event.is_directory:
            print(f"Ignoring directory: {event.src_path}", flush=True)
            return
        
        filename = os.path.basename(event.src_path)
        match = OMEGA_FRAME_PATTERN.match(filename)

        if match:
            region_number = int(match.group(1))
            frame_number = int(match.group(2))
            scan_number = int(match.group(3))
            print(f"New Omega Frame detected: Region {region_number}, Frame {frame_number}, Scan {scan_number}", flush=True)
            redis_client.sadd(f"region_{region_number}_scan_{scan_number}_frames", frame_number)
        else:
            print(f"Skipping unrecognized file: {filename}", flush=True)

def watch_for_omega_frames():
    print("Starting watchdog observer...", flush=True)
    observer = PollingObserver()  # Use PollingObserver instead of Observer()
    event_handler = OmegaFrameHandler()
    observer.schedule(event_handler, OMEGA_FRAME_DIR, recursive=False)
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
    watch_for_omega_frames()




