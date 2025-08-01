# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:33:36 2025

@author: dpqb1
"""
import sys
import os
import yaml
import argparse

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import logging
import json
import pickle
import copy
import importlib.util
import utilities as util
import numpy as np
import glob
import pandas as pd
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Constants
DEX_DATA = "/dex_data"
PARAM_DIR = "/param_files"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_environment(system_config):
    os.environ["REDIS_HOST"] = system_config["redis_host"]
    os.environ["REDIS_PORT"] = str(system_config["redis_port"])
    os.environ["DB_HOST"] = system_config["db_host"]
    os.environ["DB_NAME"] = system_config["db_name"]
    os.environ["DB_USER"] = system_config["db_user"]
    os.environ["DB_PWD"] = system_config["db_pwd"]
    redis_client = redis.Redis(host=system_config["redis_host"], port=str(system_config["redis_port"]), decode_responses=False)
    return redis_client

def wait_for_file_stable(file_path, check_interval=1.0, stable_time=3.0):
    """
    Waits until a file exists and its size remains stable for `stable_time` seconds.
    """
    last_size = -1
    stable_since = None

    while True:
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            if current_size == last_size:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_time:
                    return True
            else:
                stable_since = None
            last_size = current_size
        else:
            stable_since = None
            last_size = -1

        time.sleep(check_interval)

def load_jobs(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            jobs = json.load(f)
    else:
        raise ValueError("Unsupported job input format: must be .json")
    return jobs

def find_ff_files(ff_dir):
    """
    Finds ff1_*.h5 and ff2_*.h5 in the given directory.
    Assumes exactly one match for each.
    """
    ff1_list = glob.glob(os.path.join(ff_dir, "ff1_*.h5"))
    ff2_list = glob.glob(os.path.join(ff_dir, "ff2_*.h5"))

    if len(ff1_list) == 1 and len(ff2_list) == 1:
        return ff1_list[0], ff2_list[0]
    return None, None

def listen_for_directories(data_dir, scan_info_file, starting_scan, jobs, redis_client):
    """
    Monitors `dataDir` for new numbered subdirectories containing
    ff1_XXXXXX.h5 and ff2_XXXXXX.h5 files under ff/.
    """
    logging.info(f"Listening in {data_dir} for new Dexela directories...")
    scan = 0
    scan_n = starting_scan

    while True:
        # Load in scan information to determine next scan to process
        df = read_scan_info(scan_info_file)
        
        # Only try to process scan if it has info in file
        logging.info(f"Checking scan {scan_n}")
        scan_numbers = df['SCAN_N'].values
        if scan_n in scan_numbers:
            i = np.where(scan_n == scan_numbers)[0][0]
            # Filter by scan type
            scan_types = df['scan_type'].values
            if  not scan_types[i] in ["ff"]:
                logging.info(f"Scan number {scan_n} is not ff. Skipped")
                while True:
                    try:
                        scan_n = scan_numbers[i+1]
                        break
                    except:
                        logging.info("Waiting for .par file to update next scan")
                        time.sleep(1)
                        df = read_scan_info(scan_info_file)
                        scan_numbers = df['SCAN_N'].values
                continue
            
            # Get files
            ff_dir = os.path.join(data_dir, str(scan_n), "ff")
            file1, file2 = find_ff_files(ff_dir)
            
            if file1 == None or file2 == None:
                logging.info(f"No files present yet for scan {scan_n}")
                break

            logging.info(f"Waiting for stable files for scan {scan_n}")
            wait_for_file_stable(file1)
            wait_for_file_stable(file2)

            try:
                update_jobs(jobs, [file1,file2], scan, redis_client)
                logging.info(f"{len(jobs)} jobs generated.")
                
            except Exception as e:
                logging.error(f"Error processing scan {scan_n}: {e}")
                # Retry in next loop

            scan += 1
            # Read what next scan is going to be once the .par file updates
            while True:
                try:
                    scan_n = scan_numbers[i+1]
                    break
                except:
                    logging.info("Waiting for .par file to update next scan")
                    time.sleep(1)
                    df = read_scan_info(scan_info_file)
                    scan_numbers = df['SCAN_N'].values
                    
        time.sleep(3)

def read_scan_info(scan_info_file):
    json_file = scan_info_file + ".json"
    par_file = scan_info_file + ".par"
    # Load column name mapping from JSON
    with open(json_file, "r") as f:
        col_map = json.load(f)

    # Convert JSON keys (strings) to integers and sort
    col_names = [col_map[str(i)] for i in sorted(map(int, col_map.keys()))]

    # Load the data using whitespace as delimiter
    df = pd.read_csv(par_file, sep=r'\s+', header=None, names=col_names)
    return df
    
def update_jobs(jobs, files, scan_number,redis_client):    
    for i in range(len(jobs)):
        jobs[i]["files"] = files.copy()
        jobs[i]["scan_number"] = scan_number
        redis_client.rpush("tracking_jobs", pickle.dumps(jobs[i]))
    return jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    system_cfg = config["system"]
    detector_cfg = config["detector"]
    rings_cfg = config["rings"]

    redis_client = setup_environment(system_cfg)
    
    print("\n=== SYSTEM CONFIG ===")
    for k, v in system_cfg.items():
        print(f"{k}: {v}")

    print("\n=== DETECTOR CONFIG ===")
    for k, v in detector_cfg.items():
        print(f"{k}: {v}")

    for ring in rings_cfg:
        print(f"\n=== PROCESSING RING: {ring['name']} ===")
        print(f"  tth_deg: {ring['tth_deg']}")
        print(f"  tth_width: {ring['tth_width']}")

    # === Define paths ===
    data_dir = "/data_dir"
    param_dir = "/param_files"
    yaml_file = detector_cfg["yaml_file"]
    scan_info_file = os.path.join(data_dir, detector_cfg["scan_info_file"])

    # === Detector and ROI parameters ===
    params = {}
    params['detector'] = detector_cfg["name"]
    params['imSize'] = tuple(detector_cfg["im_size"])
    params['yamlFile'] = os.path.join(param_dir, yaml_file)
    params['start_frm'] = detector_cfg["start_frame"]

    # === Define regions of interest for each ring and generate a job for each
    region_id = 0
    jobs = []
    for ring in rings_cfg:
        # Define ring
        tth = ring["tth_deg"] * np.pi / 180
        tth_width = ring["tth_width"]
        ome_start = ring["ome_start"]
        ome_end = ring["ome_end"]
        ome_size = ring["ome_size"]
        eta_start = np.pi/180*ring["eta_start"]
        eta_end = np.pi/180*ring["eta_end"]
        num_eta_regions = ring["num_eta_regions"]
    
        eta_size = (eta_end-eta_start)/num_eta_regions
        etas = np.linspace(eta_start+eta_size/2,eta_end-eta_size/2,num_eta_regions)
        omes = np.arange(ome_start,ome_size,ome_end)
        for eta in etas:
            for ome in  omes:
                detector_distance, mm_per_pixel, _, _ = util.loadYamlData(params, tth=tth, eta=eta)
                rad = np.tan(tth) * detector_distance / mm_per_pixel
                inner_rad = rad - (tth_width - 1) / 2
                outer_rad = rad + (tth_width - 1) / 2
                deta = 1 / outer_rad
                eta_vals = np.arange(eta_start,eta_end, deta)
                num_eta = len(eta_vals)
                params['roiSize'] = [tth_width, num_eta, ome_size]

                Ainterp, new_center, x_cart, y_cart = util.getInterpParamsDexela(tth, eta, params)
                interp_params = [Ainterp, new_center, x_cart, y_cart]
        
                job = {
                "region_id": region_id,
                "scan_number": 0,
                "files": [],
                "start_frame": int(ome),
                "end_frame": int(ome+ome_size),
                "tth": float(tth),
                "eta": float(eta),
                "params": copy.deepcopy(params),
                "interp_params": copy.deepcopy(interp_params)
                }
                
                jobs.append(job)
                region_id += 1
        
    # Start directory monitoring
    starting_scan = detector_cfg["starting_scan"]
    listen_for_directories(data_dir,scan_info_file,starting_scan,jobs,redis_client)
    
if __name__ == "__main__":
    main()





