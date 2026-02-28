#!/usr/bin/env python3
# gen_feature.py
# 终极特征生成脚本：整合量价动量与真实基本面估值 (Alpha158+ Style)

import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

# ======================
# 配置
# ======================
DEFAULT_INPUT_DIR = "data/processed/combined"
DEFAULT_FEATURE_DIR = "data/features"
DEFAULT_WORKERS = min(16, max(1, cpu_count() - 1))
DEFAULT_FUTURE_DAYS = 20  
DEFAULT_LABEL_THRESHOLD = 0.05

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / series.replace(0, np.nan)

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / df['close'].replace(0, np.nan)

def compute_features_for_df(df, future_days, label_threshold):
    # 1. 基础预处理
    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 动量特征 (Momentum)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)

    # 3. 均线特征 (Moving Averages)
    for w in [5, 20, 60, 120]:
        df[f"dist_ma{w}"] = df["close"] / df["close"].rolling(w).mean().replace(0, np.nan) - 1

    # 4. 波动率与风险 (Risk)
    df["std_20"] = df["ret_1"].rolling(20).std()
    df["std_60"] = df["ret_1"].rolling(60).std()
    df["atr_14"] = calculate_atr(df, 14)
    df["skew_20"] = df["ret_1"].rolling(20).skew()
    df["kurt_20"] = df["ret_1"].rolling(20).kurt()

    # 5. 成交量特征 (Volume)
    df["vol_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean().replace(0, np.nan)
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    df["corr_cv_20"] = df["close"].rolling(20).corr(df["volume"])
    
    # VWAP 乖离 (真实成本偏离)
    # Combined 数据中 volume 是手，乘以 100 转为股
    df["vwap"] = df["amount"] / (df["volume"] * 100.0 + 1e-8)
    df["vwap_ratio"] = df["close"] / df["vwap"].replace(0, np.nan) - 1

    # 6. 技术指标 (Technical)
    df["rsi_14"] = calculate_rsi(df["close"], 14) / 100.0
    df["macd_diff"] = calculate_macd(df["close"])
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # 7. 核心基本面/风格因子 (Fundamental - NEW!)
    # 市值 (取对数)
    if 'circ_mv' in df.columns:
        df["log_mcap"] = np.log1p(df["circ_mv"].clip(lower=0))
    # PE 倒数 (E/P), PB 倒数 (B/P)
    if 'pe_ttm' in df.columns:
        df["ep_ttm"] = 1.0 / df["pe_ttm"].replace(0, np.nan)
    if 'pb' in df.columns:
        df["bp"] = 1.0 / df["pb"].replace(0, np.nan)

    # 8. 情绪特征
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["dist_high_20"] = df["close"] / df["high_20"].replace(0, np.nan) - 1
    df["dist_low_20"] = df["close"] / df["low_20"].replace(0, np.nan) - 1

    # 9. 标签 (Label)
    df[f"future_{future_days}d_ret"] = df["close"].shift(-future_days) / df["close"] - 1
    
    # 状态位
    df["is_trading"] = (df["volume"] > 0).astype("Int8")

    # 特征列表定义 (需同步至训练脚本)
    feature_cols = [
        "ret_1", "ret_5", "ret_20", "ret_60",
        "dist_ma5", "dist_ma20", "dist_ma60", "dist_ma120",
        "std_20", "std_60", "atr_14", "skew_20", "kurt_20",
        "vol_ratio_5", "vol_ratio_20", "corr_cv_20", "vwap_ratio",
        "rsi_14", "macd_diff", "hl_range",
        "log_mcap", "ep_ttm", "bp", "turnover",
        "dist_high_20", "dist_low_20"
    ]
    
    meta_cols = ["date", "symbol", "close", "amount"]
    label_cols = [f"future_{future_days}d_ret", "is_trading"]
    
    final_cols = meta_cols + feature_cols + label_cols
    return df[[c for c in final_cols if c in df.columns]]

def save_optimized_parquet(df, path):
    """通过降采样和强力压缩减少磁盘占用"""
    # 1. 浮点数降精度：float64 -> float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    # 2. 整数降精度：int64 -> int32
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    # 3. 使用 zstd 压缩保存
    df.to_parquet(path, index=False, engine='pyarrow', compression='zstd')

def process_symbol_file(path, out_dir, future_days, label_threshold, overwrite):
    try:
        df = pd.read_parquet(path)
        symbol = os.path.splitext(os.path.basename(path))[0]
        df["symbol"] = symbol

        feat = compute_features_for_df(df, future_days, label_threshold)
        out_path = os.path.join(out_dir, f"{symbol}.parquet")

        if os.path.exists(out_path) and not overwrite:
            old = pd.read_parquet(out_path)
            merged = pd.concat([old, feat], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep='last').sort_values("date").reset_index(drop=True)
            save_optimized_parquet(merged, out_path)
        else:
            save_optimized_parquet(feat, out_path)
        return (symbol, "ok", None)
    except Exception as e:
        return (path, "fail", str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--future-days", type=int, default=DEFAULT_FUTURE_DAYS)
    parser.add_argument("--label-thresh", type=float, default=DEFAULT_LABEL_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))

    print(f"Processing {len(files)} files from {args.input_dir}...")
    func = partial(process_symbol_file, out_dir=args.out_dir, future_days=args.future_days, label_threshold=args.label_thresh, overwrite=args.overwrite)

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(func, files), total=len(files)))

    ok_count = sum(1 for r in results if r[1] == "ok")
    fail_list = [r for r in results if r[1] == "fail"]
    print(f"Done. OK: {ok_count}, FAIL: {len(fail_list)}")

if __name__ == "__main__":
    main()
