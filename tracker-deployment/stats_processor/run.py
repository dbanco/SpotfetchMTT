# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:25:19 2025

@author: dpqb1
"""

import psycopg2
import json
import numpy as np
import matplotlib.pyplot as plt

DB_NAME = "tracker_data"
DB_USER = "dbanco"
DB_PASS = "yourpassword"
DB_HOST = "postgres_db"

def compute_stats():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        cursor = conn.cursor()

        query = """
        SELECT scan_number, features->'bbox_size' AS bbox_size
        FROM measurements
        WHERE detected = TRUE AND overlapping = FALSE;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        stats = {}
        for scan, bbox in rows:
            width = json.loads(bbox)[0]  # Assuming bbox_size is a JSON array
            stats.setdefault(scan, []).append(float(width))

        for scan in sorted(stats):
            widths = stats[scan]
            print(f"Scan {scan}: mean width = {np.mean(widths):.2f}, count = {len(widths)}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error computing statistics: {e}")

def compute_histogram_over_time(num_bins=50):
    conn = psycopg2.connect(
        dbname="tracker_data",
        user="dbanco",
        password="yourpassword",
        host="postgres_db"
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT scan_number, features->'variance'
        FROM measurements
        WHERE detected = TRUE AND overlapping = FALSE
        ORDER BY scan_number;
    """)
    
    rows = cursor.fetchall()
    conn.close()

    histograms = []
    scan_list = []
    all_values = []

    grouped = {}
    for scan_number, variance in rows:
        print(f"Scan:{scan_number}, Var:{variance}",flush=True)
        if variance:
            grouped.setdefault(scan_number, []).append(max(variance))
            all_values.append(max(variance))

    # Use global range for bins
    all_values = np.array(all_values)
    hist_range = (np.min(all_values), np.max(all_values))
    bins = np.linspace(*hist_range, num_bins + 1)

    for scan in sorted(grouped):
        values = grouped[scan]
        hist, _ = np.histogram(values, bins=bins)
        histograms.append(hist)
        scan_list.append(scan)

    hist_matrix = np.array(histograms).T  # Shape: (num_bins, num_scans)

    # Save to disk
    np.savez("variance_histogram.npz", hist=hist_matrix, bins=bins, scans=scan_list)

    return hist_matrix, bins, scan_list

def plot_histogram_image(hist_matrix, bins, scans):
    plt.figure(figsize=(12, 5))
    plt.imshow(hist_matrix, aspect='auto', origin='lower',
               extent=[scans[0], scans[-1], bins[0], bins[-1]])
    plt.colorbar(label='Frequency')
    plt.xlabel("Scan Number")
    plt.ylabel("Max Variance")
    plt.title("Max Variance Histogram Over Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hist_matrix, bins, scan_list = compute_histogram_over_time(num_bins=50)
    plot_histogram_image(hist_matrix, bins, scan_list)
    plt.savefig("/app/variance_histogram.png")
