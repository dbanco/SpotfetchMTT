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
        cursor = conn.cursor()

        query = """
        SELECT scan_number, (features->>'bbox_size')::jsonb->>0 AS width
        FROM measurements
        WHERE detected = TRUE AND overlapping = FALSE;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        for scan_number, width in results:
            print(f"Scan {scan_number}, Width: {width}")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Query error: {e}")
        return False

query_spot_widths_per_scan()