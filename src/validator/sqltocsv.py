#!/usr/bin/env python3
import sqlite3
import csv


def export_data(database: str, output: str):
    with sqlite3.connect(database) as con:
        cur = con.cursor()

        # Fetch all job data from the jobs table
        cur.execute("SELECT job, description FROM jobs")
        rows = cur.fetchall()

        # Write job data to a CSV file
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(["job", "description"])
            # Write the rows
            writer.writerows(rows)


if __name__ == "__main__":
    database = "jobs.db"
    output = "job_training_data.csv"
    export_data(database, output)
    print(f"Successfully exported data from {database} to {output}")
