# Scaled Deployment of Multitarget Tracking

This repository contains a real-time, scalable spot tracking pipeline for processing diffraction data across time. It supports both local Docker-based deployment and high-performance computing environments using Singularity.

---

## System Overview

This system is designed to detect, track, and analyze diffraction spots in 3D regions over time. It is capable of handling streaming data with modular services that can scale independently.

### Components

| Service             | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `omega_frame_listener` | Watches for incoming omega frame files and registers them in Redis.       |
| `job_dispatcher`    | Dispatches region-based tracking jobs when frame data is complete.          |
| `tracker`           | Runs tracking on 3D regions using MHT and saves output to PostgreSQL.       |
| `stats_processor`   | Computes and visualizes statistics based on tracking results.               |
| `redis`             | Used as a lightweight communication and job queue backend.                  |
| `postgres`          | Stores structured tracking data and supports query-based access.            |

---

## Data Flow

1. **Frame Generation**: 3D omega frames are saved as `.npy` files to a shared directory.
2. **Detection**: `omega_frame_listener` detects new files and updates Redis to record them.
3. **Job Management**: `job_dispatcher` monitors Redis and queues a job when all required frames are present.
4. **Tracking**: One of the tracker instances pulls the job, processes the region, and writes the result to PostgreSQL.
5. **Statistics**: `stats_processor` queries the database to extract statistics and optionally visualize them.

---

## Database Schema

- `regions(region_id, description)`
- `tracks(region_id, track_id, first_detected_scan, last_detected_scan)`
- `measurements(id, region_id, track_id, scan_number, detected, overlapping, features JSONB)`

Indexes exist for efficient access:
- `scan_number`
- `(region_id, track_id)`
- `detected = TRUE`
- `detected = TRUE AND overlapping = FALSE`

---

## Key Features

- **Real-time Streaming**: Processes incoming data as it's written, with minimal coordination.
- **Scalable Trackers**: Multiple independent trackers can process different regions concurrently.
- **Redis Coordination**: Redis is used to coordinate job readiness and ensure efficient job dispatching.
- **PostgreSQL Backend**: Stores rich, structured data with support for custom feature queries.
- **Singularity + SGE Compatibility**: The system is compatible with HPC environments using `.sif` containers and batch jobs.

---

## Requirements

- Python 3.10+
- Docker (for local deployment)
- Singularity/Apptainer (for HPC deployment)
- Redis and PostgreSQL
- Required Python packages (see `requirements.txt` in each subfolder)

---

## Directory Structure

