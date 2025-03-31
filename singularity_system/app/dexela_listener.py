# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:33:36 2025

@author: dpqb1
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import logging
import json
import importlib.util
import utilities as util
import numpy as np
import glob
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

REGION_DIR = "/region_files"

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
    elif path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("job_config", path)
        job_def = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_def)
        jobs = job_def.generate_jobs()
    else:
        raise ValueError("Unsupported job input format: must be .json or .py")
    return jobs

def process_files(file1, file2, tth, eta, frames, interp_params, scan):
    """
    Replace with your actual processing logic.
    """
    logging.info(f"Processing {file1} and {file2}")
    # Simulate processing time
    ring3D = util.loadDexPolarRoi3D([file1,file2], tth, eta, frames, params, 
                                    interp_params=interp_params)
    # Load in jobs json
    jobs = load_jobs("job_config.json")
    
    for job in jobs:
        region_id = job["region_id"]
        eta_inds = job["eta_inds"]
        start_frame = job["start_frame"]
        end_frame = job["end_frame"]
        roi = ring3D[:,eta_inds[0]:eta_inds[1]+1,start_frame:end_frame+1]
        outFile = os.path.join(REGION_DIR,f'region_{region_id}_scan_{scan}')
        np.save(outFile,roi)

    logging.info("Processing complete.")

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

def listen_for_dexela_directories(data_dir, tth, eta, frames, interp_params, scan, scan_info_file, starting_scan):
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
            
            # Determine regions to process
            
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
                process_files(file1, file2, tth, eta, frames, interp_params, scan)

            except Exception as e:
                logging.error(f"Error processing scan {scan_n}: {e}")
                # Retry in next loop

            scan += 1
            # Read what next scan is going to be 
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

# def determine_regions(df,roi_size,num_eta_regions,eta_vals,tth,output_path):
#     ome_start = df['ome_start_real']
#     ome_end = df['ome_end_real']

# def generate_jobs(roi_size,total_ome,num_eta_regions,eta_vals,tth,output_path,scan):
#     region_id = 0
#     jobs = []
    
#     total_eta = roi_size[1]
#     ome_width = roi_size[2]
    
#     eta_width = int(round(total_eta/num_eta_regions))
    
#     start_eta = 0   

#     while start_eta < total_eta:
#         print(start_eta)
#         for start_ome in np.arange(0,total_ome-ome_width+1,ome_width):
            
#             start_frame = start_ome
#             end_frame = start_ome + ome_width - 1
            
#             eta_min = eta_vals[start_eta]
#             eta_max_ind = min(start_eta + eta_width - 1, total_eta-1)
#             eta_max = eta_vals[eta_max_ind]
            
#             jobs.append({
#                 "region_id": region_id,
#                 "scan": scan,
#                 "start_frame": int(start_frame),
#                 "end_frame": int(end_frame),
#                 "eta_inds": [int(start_eta), int(eta_max_ind)],
#                 "eta": [float(eta_min), float(eta_max)],
#                 "tth": float(tth),
#             })
        
#             region_id += 1
#         start_eta += eta_width
#     with open(output_path, "w") as f:
#         json.dump(jobs, f, indent=2)
#     print(f"Wrote {len(jobs)} jobs to {output_path}")
    
def generate_jobs(roi_size,total_ome,ome_width,num_eta_regions,eta_vals,tth,output_path):
    region_id = 0
    jobs = []
    
    total_eta = roi_size[1]
    
    eta_width = int(round(total_eta/num_eta_regions))
    
    start_eta = 0   

    while start_eta < total_eta:
        print(start_eta)
        for start_ome in np.arange(0,total_ome-ome_width+1,ome_width):
            
            start_frame = start_ome
            end_frame = min(start_ome + ome_width - 1, total_ome-1)
            
            eta_min = eta_vals[start_eta]
            eta_max_ind = min(start_eta + eta_width - 1, total_eta-1)
            eta_max = eta_vals[eta_max_ind]
            
            jobs.append({
                "region_id": region_id,
                "scan_number": 0,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "eta_inds": [int(start_eta), int(eta_max_ind)],
                "eta": [float(eta_min), float(eta_max)],
                "tth": float(tth),
            })
        
            region_id += 1
        start_eta += eta_width
    with open(output_path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"Wrote {len(jobs)} jobs to {output_path}")

if __name__ == "__main__":
    # topPath = "/nfs/chess/user/dbanco/c103_processing"
    # dex_dir = "/nfs/chess/user/dbanco/SpotfetchMTT/tracker-deploy-chess/dex_data_test"
    # dex_dir = r"/nfs/chess/raw/2024-2/id3a/miller-3528-c/c103-1-ff-1"
    # params['yamlFile'] = os.path.join(topPath,"dexelas_calibrated_ruby_0504_v01.yml")
    
    
    dex_data = "/dex_data"
    param_dir = "/param_files"
    
    params = {}
    params['detector'] = 'dexela'
    params['imSize'] = (4000,6500)
    params['yamlFile'] = os.path.join(param_dir,'ceo2_dexela_instr_032025.yml')
    params['start_frm'] = 0
    
    scan = 0
    eta = 0
    tth = 3.63*np.pi/180 #degrees
    tth_width = 35 #pixels
    ome_width = 10 #frames
    detector_distance, mm_per_pixel, ff_trans, ff_tilt = util.loadYamlData(params,tth=tth,eta = 0)
    
    rad = np.tan(tth)*detector_distance/mm_per_pixel
    inner_rad = rad - (tth_width-1)/2
    outer_rad = rad + (tth_width-1)/2
    deta = 1/outer_rad
    right_eta_vals = np.arange(-0.8*np.pi/2,0.8*np.pi/2,deta)
    num_right_eta = len(right_eta_vals)
    
    num_eta_regions = 18
    total_ome = 40
    frames = [0,total_ome-1]
    
    params['roiSize'] = [tth_width,num_right_eta,total_ome]
    Ainterp, new_center, x_cart, y_cart = util.getInterpParamsDexela(tth, eta, params)
    interp_params = []
    interp_params.append(Ainterp)
    interp_params.append(new_center)
    interp_params.append(x_cart)
    interp_params.append(y_cart)
    
    job_config_file = "job_config.json"
    generate_jobs(params['roiSize'],total_ome,ome_width,num_eta_regions,right_eta_vals,tth,job_config_file)

    # params['roiSize'][2] = 8
    # frames = [0,7]
    # generate_jobs(params['roiSize'],8,num_eta_regions,right_eta_vals,tth,output_path)
    
    starting_scan = 14
    scan_info_file = os.path.join(dex_data,"rams2-slew_ome")
    listen_for_dexela_directories(dex_data,tth,eta,frames,interp_params,scan,scan_info_file,starting_scan)


    # x1 = inner_rad*np.cos(right_eta_vals) + params['imSize'][1]/2;
    # x2 = inner_rad*np.cos(right_eta_vals) + params['imSize'][1]/2;
    # y1 = outer_rad*np.sin(right_eta_vals) + params['imSize'][0]/2;
    # y2 = outer_rad*np.sin(right_eta_vals) + params['imSize'][0]/2;

    # plt.figure(figsize=(50, 20))
    # plt.imshow(img,vmax=500)
    # plt.plot(x1,y1,'-')
    # plt.plot(x2,y2,'-')
    # plt.show()

    
    # ring = util.loadPolarROI(fnames, tth, 0, 100, params)




