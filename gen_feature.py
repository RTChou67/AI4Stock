#!/usr/bin/env python3
# gen_feature.py
# 增强版特征生成脚本：引入更多稳健的技术指标 (RSI, ATR, MACD, etc.)
# 修改 Label：使用风险调整后收益 (Sharpe Ratio Proxy)

import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

# 默认配置
DEFAULT_DAILY_DIR = "data/daily"
DEFAULT_FEATURE_DIR = "data/features"
DEFAULT_WORKERS = min(8, max(1, cpu_count() - 1))
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
    return (macd_line - signal_line) / series

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / df['close']

def compute_features_for_df(df, future_days, label_threshold):
    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 基础收益
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)

    # 均线乖离率
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()
    
    df["dist_ma5"] = df["close"] / df["ma5"] - 1
    df["dist_ma20"] = df["close"] / df["ma20"] - 1
    df["dist_ma60"] = df["close"] / df["ma60"] - 1
    df["dist_ma120"] = df["close"] / df["ma120"] - 1

    # 波动率
    df["std_20"] = df["ret_1"].rolling(20).std()
    df["std_60"] = df["ret_1"].rolling(60).std()
    
    # ATR
    df["atr_14"] = calculate_atr(df, 14)

    # 量价因子
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio_5"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)
    df["vol_ratio_20"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # RSI
    df["rsi_14"] = calculate_rsi(df["close"], 14) / 100.0

    # MACD
    df["macd_diff"] = calculate_macd(df["close"])

    # High-Low Range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Sentiment
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["dist_high_20"] = df["close"] / df["high_20"] - 1
    df["dist_low_20"] = df["close"] / df["low_20"] - 1

    # ==== 关键修改：计算风险调整后收益 (Label) ====
    
    # 1. 原始未来收益
    raw_future_ret = df["close"].shift(-future_days) / df["close"] - 1
    
    # 2. 未来波动率 (使用未来窗口内的日收益率标准差)
    # shift(-future_days) 是窗口结束点。我们想算从 T+1 到 T+future_days 的 std。
    # pandas rolling(window).std().shift(-window) 可以实现
    future_std = df["ret_1"].rolling(window=future_days).std().shift(-future_days)
    
    # 3. 计算 Sharpe Proxy (避免除以0，加上一个小常数)
    # 限制极值，避免离群点
    df[f"future_{future_days}d_sharpe"] = (raw_future_ret / (future_std + 1e-4)).clip(-5, 5)
    
    # 保留原始 return 用于回测计算真实收益
    df[f"future_{future_days}d_ret"] = raw_future_ret
    
    df["is_trading"] = (df["volume"] > 0).astype("Int8")

    feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
        "dist_ma5", "dist_ma20", "dist_ma60", "dist_ma120",
        "std_20", "std_60", "atr_14",
        "vol_ratio_5", "vol_ratio_20",
        "rsi_14", "macd_diff", "hl_range",
        "dist_high_20", "dist_low_20"
    ]
    
    meta_cols = ["date", "symbol", "close", "amount"]
    # 两个 future 列都保留
    label_cols = [f"future_{future_days}d_ret", f"future_{future_days}d_sharpe", "is_trading"]
    
    final_cols = meta_cols + feature_cols + label_cols
    return df[[c for c in final_cols if c in df.columns]]


def process_symbol_file(path, out_dir, future_days, label_threshold, overwrite):
    try:
        df = pd.read_parquet(path)
        if "symbol" not in df.columns:
            base = os.path.basename(path)
            sym = os.path.splitext(base)[0]
            df["symbol"] = sym

        feat = compute_features_for_df(df, future_days, label_threshold)

        out_sym = feat["symbol"].iloc[0]
        out_path = os.path.join(out_dir, f"{out_sym}.parquet")

        if os.path.exists(out_path) and not overwrite:
            old = pd.read_parquet(out_path)
            merged = pd.concat([old, feat], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep='last').sort_values("date").reset_index(drop=True)
            merged.to_parquet(out_path, index=False)
        else:
            feat.to_parquet(out_path, index=False)

        return (out_sym, "ok", None)
    except Exception as e:
        return (path, "fail", str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-dir", type=str, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--future-days", type=int, default=DEFAULT_FUTURE_DAYS)
    parser.add_argument("--label-thresh", type=float, default=DEFAULT_LABEL_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.daily_dir, "*.parquet")))
    if args.limit: files = files[: args.limit]

    print(f"Processing {len(files)} files with {args.workers} workers...")
    print("New Label: future_{}d_sharpe (Risk Adjusted)".format(args.future_days))

    func = partial(
        process_symbol_file, 
        out_dir=args.out_dir, 
        future_days=args.future_days, 
        label_threshold=args.label_thresh, 
        overwrite=args.overwrite
    )

    results = []
    with Pool(args.workers) as p:
        for res in tqdm(p.imap_unordered(func, files), total=len(files)):
            results.append(res)

    ok_count = sum(1 for r in results if r[1] == "ok")
    fail_list = [r for r in results if r[1] == "fail"]
    
    print(f"Done. OK: {ok_count}, FAIL: {len(fail_list)}")


if __name__ == "__main__":
    main()
