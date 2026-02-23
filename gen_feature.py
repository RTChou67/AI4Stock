#!/usr/bin/env python3
# generate_features_per_symbol.py
# 说明：
#  - 输入：data/daily/{symbol}.parquet （由 akshare 并发抓取脚本生成）
#  - 输出：data/features/{symbol}.parquet （每个文件包含该 symbol 的所有滚动特征与未来标签）
#  - 并行处理：默认使用多进程（POOL_WORKERS），可在环境变量或命令行调整

import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from functools import partial
from tqdm import tqdm

# 配置（可由命令行覆盖）
DATA_DAILY_DIR = "data/daily"
DATA_FEATURE_DIR = "data/features"
POOL_WORKERS = min(8, max(1, cpu_count() - 1))   # 默认并发数，Linux 上可调整
FUTURE_DAYS = 5
LABEL_THRESHOLD = 0.02   # 5-day 回报超过 2% 记为正类

os.makedirs(DATA_FEATURE_DIR, exist_ok=True)

def compute_features_for_df(df):
    """
    输入单个symbol的日线 DataFrame（必须包含列: date, close, open, high, low, volume, amount, turnover）
    返回同长度 DataFrame，带上新增特征与 future label
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 基础收益（simple pct change）
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
    df["vol_z20"] = (df["volume"] - df["vol_ma20"]) / (df["volume"].rolling(20, min_periods=5).std().replace(0, pd.NA))

    # Liquidity
    if "turnover" in df.columns:
        df["turnover_ma20"] = df["turnover"].rolling(window=20, min_periods=5).mean()
    else:
        df["turnover_ma20"] = pd.NA

    # Price-level flags
    # 新高 250 天
    df["new_high_250"] = df["close"].rolling(window=250, min_periods=1).apply(lambda x: 1 if x.iloc[-1] >= x.max() else 0)

    # Gaps: 昨日收盘到今日开盘 gap
    df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # 未来收益标签（future_n days）
    df[f"future_{FUTURE_DAYS}d_ret"] = df["close"].shift(-FUTURE_DAYS) / df["close"] - 1
    df[f"label_{FUTURE_DAYS}d_bin"] = (df[f"future_{FUTURE_DAYS}d_ret"] > LABEL_THRESHOLD).astype("Int8")

    # 标注是否可交易（volume>0）
    df["is_trading"] = (df["volume"] > 0).astype("Int8")

    # 最后清理：保留常用列顺序
    keep_cols = [
        "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover",
        "r1", "r5", "r20", "ma5", "ma10", "ma20", "mom20",
        "vol5", "vol20", "vol_ma20", "vol_z20", "turnover_ma20",
        "new_high_250", "overnight_gap",
        f"future_{FUTURE_DAYS}d_ret", f"label_{FUTURE_DAYS}d_bin", "is_trading"
    ]
    # 只保留存在的列
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def process_symbol_file(path, future_days=FUTURE_DAYS, label_thresh=LABEL_THRESHOLD, overwrite=False):
    """
    处理一个 symbol 的 parquet 文件
    """
    try:
        # 读取
        df = pd.read_parquet(path)
        # 规范列名：某些 akshare 返回列名有轻微差别，统一小写 key checks
        # 假设抓取脚本已规范化为 'date','symbol','open','high','low','close','volume','amount','turnover'
        # 如果 symbol 列缺失，则从文件名中读取
        if "symbol" not in df.columns:
            # filename like .../000001.parquet
            base = os.path.basename(path)
            sym = os.path.splitext(base)[0]
            df["symbol"] = sym

        # 计算特征
        feat = compute_features_for_df(df)

        out_sym = feat["symbol"].iloc[0]
        out_path = os.path.join(DATA_FEATURE_DIR, f"{out_sym}.parquet")

        if os.path.exists(out_path) and not overwrite:
            # 合并：如果已有历史文件，取 union（去重日期）
            old = pd.read_parquet(out_path)
            merged = pd.concat([old, feat], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            merged.to_parquet(out_path, index=False)
        else:
            feat.to_parquet(out_path, index=False)

        return (out_sym, "ok", None)
    except Exception as e:
        return (path, "fail", str(e))


def main(args):
    # 列出所有 daily parquet
    files = sorted(glob.glob(os.path.join(args.daily_dir, "*.parquet")))
    if args.limit:
        files = files[: args.limit]

    worker = args.workers or POOL_WORKERS
    print(f"Found {len(files)} files, processing with {worker} workers")

    func = partial(process_symbol_file, overwrite=args.overwrite)
    results = []
    with Pool(worker) as p:
        for res in tqdm(p.imap_unordered(func, files), total=len(files)):
            results.append(res)

    ok = sum(1 for r in results if r[1] == "ok")
    fail = [r for r in results if r[1] == "fail"]
    print(f"Done. OK: {ok}, FAIL: {len(fail)}")
    if fail:
        print("Failures (sample):", fail[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-dir", type=str, default=DATA_DAILY_DIR, help="input daily parquet dir")
    parser.add_argument("--out-dir", type=str, default=DATA_FEATURE_DIR, help="output features dir (ignored; set in script)")
    parser.add_argument("--workers", type=int, default=None, help="num parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="limit number of symbols processed (for debug)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing feature files")
    args = parser.parse_args()

    DATA_DAILY_DIR = args.daily_dir
    DATA_FEATURE_DIR = args.out_dir
    os.makedirs(DATA_FEATURE_DIR, exist_ok=True)

    main(args)
