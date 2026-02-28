#!/usr/bin/env python3
"""
一键数据瘦身工具 (Data Slimming Tool)
遍历 data/ 目录下的所有 Parquet 文件，
将其转换为 float32/int32 并使用 zstd 强力压缩。
"""

import os
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

DATA_DIRS = ["data/raw/daily", "data/raw/valuation", "data/processed/combined", "data/features"]

def slim_file(path):
    try:
        df = pd.read_parquet(path)
        original_size = os.path.getsize(path)
        
        # 1. 降采样
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
            
        # 2. 覆盖保存 (zstd)
        df.to_parquet(path, index=False, engine='pyarrow', compression='zstd')
        
        new_size = os.path.getsize(path)
        return (path, original_size, new_size)
    except Exception as e:
        return (path, 0, 0, str(e))

def main():
    files = []
    for d in DATA_DIRS:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*.parquet")))
    
    print(f"Total files to process: {len(files)}")
    if not files: return

    workers = cpu_count()
    results = []
    with Pool(workers) as p:
        for res in tqdm(p.imap_unordered(slim_file, files), total=len(files), desc="Compressing"):
            results.append(res)

    # 统计成果
    total_original = sum(r[1] for r in results)
    total_new = sum(r[2] for r in results)
    
    print("\n" + "="*40)
    print("SLIMMING COMPLETE")
    print(f"Original Size: {total_original / 1024 / 1024:.2f} MB")
    print(f"New Size     : {total_new / 1024 / 1024:.2f} MB")
    if total_original > 0:
        reduction = (1 - total_new / total_original) * 100
        print(f"Space Saved  : {reduction:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
