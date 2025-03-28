# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 23:06:20 2025

@author: dpqb1
"""

import psycopg2
import json

# Database connection settings
DB_NAME = "tracker_data"
DB_USER = "dbanco"
DB_PASS = "yourpassword"
DB_HOST = "postgres_db"

def query_spot_widths_per_scan():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        conn.autocommit = True
        cursor = conn.cursor()

        query = """
        INSERT INTO regions (region_id)
        VALUES (%s)
        ON CONFLICT DO NOTHING;
        """
        
        region_id = 0
        track_id = 0
        # cursor.execute(query, (region_id,))

        query = """
        INSERT INTO tracks (region_id, track_id, first_detected_scan)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        
        # cursor.execute(query, (region_id, track_id, 1))


        features = {
            "com": 0,  # Convert NumPy arrays to lists
            "velocity": 0,
            "bbox": 0,
            "bbox_center": 20,
            "bbox_size": 10,
            "intensity": 200,
            "principal_axes": [1,2,3],
            "variance": [4,5,6]
        }
        
        query = """
        INSERT INTO measurements (region_id, track_id, scan_number, detected, overlapping, features)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (region_id, track_id, scan_number) DO UPDATE
        SET detected = EXCLUDED.detected, features = EXCLUDED.features;
        """

        # Execute query
        # cursor.execute(query, (region_id, track_id, 0, True, False, json.dumps(features)))
        # conn.commit()

        print(conn, flush=True)

        query = """
        SELECT *
        FROM measurements
        WHERE detected = TRUE AND overlapping = FALSE
        ORDER BY region_id, track_id, scan_number;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        print("Query returned rows:", len(results), flush=True)

        for row in results:
            print(row, flush=True)

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Query error: {e}", flush=True)
        return False

    
if __name__ == "__main__":
    
    print("PLEASE",flush=True)
    query_spot_widths_per_scan()