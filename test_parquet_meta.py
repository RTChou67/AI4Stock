import pyarrow.parquet as pq
import sys
import pandas as pd

file_path = "data/daily/000001.parquet"
try:
    meta = pq.read_metadata(file_path)
    print("Num rows:", meta.num_rows)
    schema = meta.schema
    print("Columns:", schema.names)
    
    rg = meta.row_group(0)
    for i in range(rg.num_columns):
        col_meta = rg.column(i)
        print(f"Col {schema.names[i]}: has stats={col_meta.is_stats_set}, nulls={col_meta.statistics.null_count if col_meta.is_stats_set else 'unknown'}")
        if schema.names[i] == 'date' and col_meta.is_stats_set:
            print(f"  Min date: {col_meta.statistics.min}, Max date: {col_meta.statistics.max}")
except Exception as e:
    print("Error:", e)
