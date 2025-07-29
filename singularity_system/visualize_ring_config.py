import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import util  # assumes util.loadYamlData exists and is compatible

# Load config
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
system = cfg["system"]
detector_cfg = cfg["detector"]
rings_cfg = cfg["rings"]

# Load sample image (for visualization)
dex_data = detector_cfg["dex_data_dir"]
first_scan = detector_cfg["starting_scan"]
image_path = os.path.join(dex_data, f"output_j{first_scan}_1_1.mat")

# NOTE: Replace this with actual image loading
# If MAT file, use scipy.io.loadmat, or replace with dummy data
img = np.ones(detector_cfg["im_size"]) * 100  # placeholder white image

# Detector and calibration params
param_dir = detector_cfg["detect_yaml_dir"]
yaml_file = detector_cfg["yaml_file"]
params = {
    'detector': detector_cfg["name"],
    'imSize': tuple(detector_cfg["im_size"]),
    'yamlFile': os.path.join(param_dir, yaml_file),
    'start_frm': 0
}

detector_distance, mm_per_pixel, ff_trans, ff_tilt = util.loadYamlData(params, tth=0, eta=0)

# Plot image and overlay all rings
plt.figure(figsize=(50, 20))
plt.imshow(img, vmax=500, cmap='gray')

for ring in rings_cfg:
    tth = ring["tth_deg"] * np.pi / 180
    tth_width = ring["tth_width"]

    rad = np.tan(tth) * detector_distance / mm_per_pixel
    inner_rad = rad - (tth_width - 1) / 2
    outer_rad = rad + (tth_width - 1) / 2
    deta = 1 / outer_rad
    right_eta_vals = np.arange(-0.8 * np.pi / 2, 0.8 * np.pi / 2, deta)

    x1 = inner_rad * np.cos(right_eta_vals) + params['imSize'][1] / 2
    y1 = inner_rad * np.sin(right_eta_vals) + params['imSize'][0] / 2
    x2 = outer_rad * np.cos(right_eta_vals) + params['imSize'][1] / 2
    y2 = outer_rad * np.sin(right_eta_vals) + params['imSize'][0] / 2

    plt.plot(x1, y1, '-', label=f"{ring['name']} inner")
    plt.plot(x2, y2, '-', label=f"{ring['name']} outer")

plt.legend()
plt.title("Overlay of Ring Boundaries")
plt.show()
