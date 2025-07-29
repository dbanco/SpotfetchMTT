#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:43:28 2025

@author: dbanco
"""

import numpy as np
import json
import os
import argparse
import utilities as util

def generate_jobs(yaml_file, im_size, tth_list, roi_size, eta_bins, omega_range, omega_step, start_scan, output_path):
    yaml_data = util.read_yaml(yaml_file)
    mm_per_pixel = yaml_data['detectors']['ff2']['pixels']['size'][0]
    detector_distance = yaml_data['detectors']['ff1']['transform']['translation'][2]

    region_id = 0
    jobs = []

    for tth in tth_list:
        radius = np.tan(tth) * detector_distance / mm_per_pixel
        for i in range(eta_bins):
            eta_min = i * 2 * np.pi / eta_bins
            eta_max = (i + 1) * 2 * np.pi / eta_bins
            eta_center = (eta_min + eta_max) / 2

            for omega in np.arange(omega_range[0], omega_range[1] + 1e-6, omega_step):
                start_frame = int(omega - (roi_size[2] - 1) / 2)
                end_frame = int(omega + (roi_size[2] - 1) / 2)

                jobs.append({
                    "region_id": region_id,
                    "scan_number": start_scan,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "eta": [float(eta_min), float(eta_max)],
                    "tth": float(tth)
                })

                region_id += 1

    with open(output_path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"✅ Wrote {len(jobs)} jobs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Detector YAML file")
    parser.add_argument("--tth", nargs="+", type=float, required=True, help="List of tth values (radians)")
    parser.add_argument("--eta_bins", type=int, default=60, help="Number of η bins")
    parser.add_argument("--omega_range", nargs=2, type=int, default=[0, 179], help="Omega range in frames")
    parser.add_argument("--omega_step", type=int, default=10, help="Step in omega (for one job per step)")
    parser.add_argument("--roi_size", nargs=3, type=int, default=[30, 30, 11], help="ROI size [x, y, z]")
    parser.add_argument("--scan_number", type=int, default=0)
    parser.add_argument("--im_size", nargs=2, type=int, default=[4000, 6400])
    parser.add_argument("--out", default="tracking_jobs_config.json")

    args = parser.parse_args()

    generate_jobs(
        yaml_file=args.yaml,
        im_size=args.im_size,
        tth_list=args.tth,
        roi_size=args.roi_size,
        eta_bins=args.eta_bins,
        omega_range=args.omega_range,
        omega_step=args.omega_step,
        start_scan=args.scan_number,
        output_path=args.out
    )
