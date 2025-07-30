import yaml
import sys
import os

# Read config file from command-line argument
if len(sys.argv) < 2:
    print("Usage: python export_env.py <config.yaml>")
    sys.exit(1)

yaml_file = sys.argv[1]

with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

sys_cfg = config["system"]
detect_cfg = config["detector"]
base_dir = sys_cfg["base_dir"]

print(f'export SYS_DIR="{base_dir}"')
print(f'export REDIS_HOST="{sys_cfg["redis_host"]}"')
print(f'export REDIS_PORT={sys_cfg["redis_port"]}')
print(f'export POSTGRES_HOST="{sys_cfg["postgres_host"]}"')
print(f'export USER="{sys_cfg["user"]}"')

print(f'export SIF_DIR="{os.path.join(base_dir, sys_cfg["sif_dir"])}"')
print(f'export APP_DIR="{os.path.join(base_dir, sys_cfg["app_dir"])}"')
print(f'export DB_NAME="{sys_cfg["db_name"]}"')
print(f'export POSTGRES_DIR={os.path.join(base_dir, sys_cfg["postgres_dir"])}')
print(f'export REGION_DIR={os.path.join(base_dir, sys_cfg["region_dir"])}')
print(f'export TRACKER_STATE_DIR={os.path.join(base_dir, sys_cfg["tracker_state_dir"])}')
print(f'export NUM_TRACKERS={sys_cfg["num_trackers"]}')

print(f'export DETECT_YAML_DIR={detect_cfg["yaml_dir"]}')
print(f'export DATA_DIR={detect_cfg["data_dir"]}')



