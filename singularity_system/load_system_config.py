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
mtt_dir = sys_cfg["mtt_dir"]
sing_dir = sys_cfg["singularity_dir"]

print(f'export MTT_DIR="{mtt_dir}"')
print(f'export SING_DIR="{os.path.join(mtt_dir, sing_dir)}"')
print(f'export REDIS_HOST="{sys_cfg["redis_host"]}"')
print(f'export REDIS_PORT={sys_cfg["redis_port"]}')
print(f'export DB_HOST="{sys_cfg["db_host"]}"')
print(f'export DB_NAME="{sys_cfg["db_name"]}"')
print(f'export DB_USER="{sys_cfg["db_user"]}"')
print(f'export DB_PWD="{sys_cfg["db_pwd"]}"')

print(f'export SIF_DIR="{os.path.join(mtt_dir, sing_dir, sys_cfg["sif_dir"])}"')
print(f'export APP_DIR="{os.path.join(mtt_dir, sing_dir, sys_cfg["app_dir"])}"')
print(f'export POSTGRES_DIR={os.path.join(mtt_dir, sing_dir, sys_cfg["postgres_dir"])}')
print(f'export REGION_DIR={os.path.join(mtt_dir, sing_dir, sys_cfg["region_dir"])}')
print(f'export TRACKER_STATE_DIR={os.path.join(mtt_dir, sing_dir, sys_cfg["tracker_state_dir"])}')
print(f'export NUM_TRACKERS={sys_cfg["num_trackers"]}')

print(f'export YAML_DIR={detect_cfg["yaml_dir"]}')
print(f'export DATA_DIR={detect_cfg["data_dir"]}')



