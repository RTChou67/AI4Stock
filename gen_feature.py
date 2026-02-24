#!/usr/bin/env python3
# generate_features_per_symbol.py
# 说明：
#  - 输入：data/daily/{symbol}.parquet
#  - 输出：data/features/{symbol}.parquet
#  - 并行处理：使用 multiprocessing.Pool

import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from functools import partial
from tqdm import tqdm

# 默认配置
DEFAULT_DAILY_DIR = "data/daily"
DEFAULT_FEATURE_DIR = "data/features"
DEFAULT_WORKERS = min(8, max(1, cpu_count() - 1))
DEFAULT_FUTURE_DAYS = 20  # 统一改为20天
DEFAULT_LABEL_THRESHOLD = 0.05  # 20天回报超过 5% 记为正类 (原2%对应5天)

def compute_features_for_df(df, future_days, label_threshold):
    """
    输入单个symbol的日线 DataFrame
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 基础收益
    df["r1"] = df["close"].pct_change(1)
    df["r5"] = df["close"].pct_change(5)
    df["r20"] = df["close"].pct_change(20)

    # MA / momentum
    df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(window=20, min_periods=5).mean()
    df["mom20"] = df["close"] / df["close"].shift(20) - 1

    # Volatility
    df["vol20"] = df["r1"].rolling(window=20, min_periods=10).std()
    df["vol5"] = df["r1"].rolling(window=5, min_periods=3).std()

    # Volume related
    df["vol_ma20"] = df["volume"].rolling(window=20, min_periods=5).mean()
    # 避免除以0
    vol_std = df["volume"].rolling(20, min_periods=5).std()
    df["vol_z20"] = (df["volume"] - df["vol_ma20"]) / vol_std.replace(0, pd.NA)

    # Liquidity
    if "turnover" in df.columns:
        df["turnover_ma20"] = df["turnover"].rolling(window=20, min_periods=5).mean()
    else:
        df["turnover_ma20"] = pd.NA

    # Price-level flags
    # 新高 250 天
    df["new_high_250"] = df["close"].rolling(window=250, min_periods=1).apply(lambda x: 1 if x.iloc[-1] >= x.max() else 0)

    # Gaps
    df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # 未来收益标签（future_n days）
    df[f"future_{future_days}d_ret"] = df["close"].shift(-future_days) / df["close"] - 1
    df[f"label_{future_days}d_bin"] = (df[f"future_{future_days}d_ret"] > label_threshold).astype("Int8")

    # 标注是否可交易
    df["is_trading"] = (df["volume"] > 0).astype("Int8")

    # 保留列
    keep_cols = [
        "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover",
        "r1", "r5", "r20", "ma5", "ma10", "ma20", "mom20",
        "vol5", "vol20", "vol_ma20", "vol_z20", "turnover_ma20",
        "new_high_250", "overnight_gap",
        f"future_{future_days}d_ret", f"label_{future_days}d_bin", "is_trading"
    ]
    return df[[c for c in keep_cols if c in df.columns]]


def process_symbol_file(path, out_dir, future_days, label_threshold, overwrite):
    """
    处理一个 symbol 的 parquet 文件
    """
    try:
        df = pd.read_parquet(path)
        
        # 补全 symbol 列
        if "symbol" not in df.columns:
            base = os.path.basename(path)
            sym = os.path.splitext(base)[0]
            df["symbol"] = sym

        # 计算特征
        feat = compute_features_for_df(df, future_days, label_threshold)

        out_sym = feat["symbol"].iloc[0]
        out_path = os.path.join(out_dir, f"{out_sym}.parquet")

        if os.path.exists(out_path) and not overwrite:
            # 合并逻辑：读取旧文件，合并新数据，去重
            old = pd.read_parquet(out_path)
            # 简单去重策略：保留最新的记录
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
    if args.limit:
        files = files[: args.limit]

    print(f"Processing {len(files)} files with {args.workers} workers...")
    print(f"Config: Future={args.future_days}d, LabelThresh={args.label_thresh}")

    # 使用 partial 绑定参数，确保子进程能获取到正确配置
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
    if fail_list:
        print("Failures (first 10):")
        for f in fail_list[:10]:
            print(f"  {f[0]}: {f[2]}")


if __name__ == "__main__":
    main()
