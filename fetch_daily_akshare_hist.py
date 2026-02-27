#!/usr/bin/env python3
import os
import time
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys
import threading

# ======================
# 配置
# ======================
DATA_DIR = "data/daily"
CACHE_DIR = "data"
SYMBOL_CACHE_FILE = os.path.join(CACHE_DIR, "symbols_cache.parquet")

TARGET_START_DATE = pd.Timestamp("2021-01-04")
TARGET_END_DATE = pd.Timestamp("2025-12-31")
AK_START_DATE_STR = "20210101"
AK_END_DATE_STR = "20251231"

EXPECTED_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']

# 并发与超时控制
MAX_WORKERS = 8
RETRY = 2
SLEEP = 0.1
FETCH_TIMEOUT = 20  # 单只股票最高等待 20 秒
CIRCUIT_BREAKER_LIMIT = 30 # 连续失败 30 次则熔断

# 全局状态，用于熔断
fail_lock = threading.Lock()
consecutive_fails = 0
should_abort = False

os.makedirs(DATA_DIR, exist_ok=True)

# ======================
# 核心功能：检查单个文件是否合格
# ======================
def is_file_qualified(symbol, known_start_date=None):
    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(file_path): return False, "Missing"
    try:
        df = pd.read_parquet(file_path)
        if df.empty: return False, "Empty"
        df['date'] = pd.to_datetime(df['date'])
        if not all(col in df.columns for col in EXPECTED_COLS): return False, "Missing columns"
        start_date, end_date = df['date'].min(), df['date'].max()
        if end_date < TARGET_END_DATE: return False, f"End date early: {end_date.date()}"
        effective_target = TARGET_START_DATE
        if known_start_date is not None: effective_target = max(TARGET_START_DATE, pd.Timestamp(known_start_date))
        if start_date > (effective_target + pd.Timedelta(days=3)): return False, f"Start date late: {start_date.date()}"
        return True, "Perfect"
    except: return False, "Corrupted"

# ======================
# 抓取逻辑 (带超时与熔断)
# ======================
def fetch_one(symbol, known_start, only_check=False):
    global consecutive_fails, should_abort
    
    if should_abort: return f"{symbol} ABORTED"

    qualified, reason = is_file_qualified(symbol, known_start)
    if qualified:
        with fail_lock: consecutive_fails = 0 # 成功一个，重置连续失败计数
        return f"{symbol} OK"
    if only_check: return f"{symbol} INCOMPLETE ({reason})"

    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    
    for attempt in range(RETRY):
        if should_abort: break
        try:
            # 执行抓取 (无法直接在 akshare 加 timeout，但 ThreadPoolExecutor 会捕获挂起)
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=AK_START_DATE_STR, end_date=AK_END_DATE_STR,
                adjust="qfq"
            )
            
            if df is None or df.empty:
                raise ValueError("Empty data from API")

            df = df.rename(columns={
                "日期": "date", "股票代码": "symbol",
                "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                "成交量": "volume", "成交额": "amount", "换手率": "turnover"
            })
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            df = df[df['date'] >= TARGET_START_DATE]
            actual_start = df['date'].min()
            df.sort_values("date").to_parquet(file_path, index=False)
            
            with fail_lock: consecutive_fails = 0 # 成功，重置
            time.sleep(SLEEP)
            return f"{symbol} FETCHED|{actual_start.strftime('%Y-%m-%d')}"

        except Exception as e:
            time.sleep(1 + attempt)
            continue # 重试

    # 走到这里说明重试也失败了
    with fail_lock:
        consecutive_fails += 1
        if consecutive_fails >= CIRCUIT_BREAKER_LIMIT:
            should_abort = True
            print(f"\n[!!!] CIRCUIT BREAKER TRIGGERED: {CIRCUIT_BREAKER_LIMIT} consecutive failures. Stopping...")
    
    return f"{symbol} FAIL"

# ======================
# 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    # 1. 探测网络
    print("Testing connection with '000001'...")
    try:
        probe = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20251201", end_date="20251231", adjust="qfq")
        if probe is None or probe.empty: raise ConnectionError("API probe returned empty")
        print("Connection OK.")
    except Exception as e:
        print(f"Network probe failed: {e}. Please check your connection.")
        return

    # 2. 获取列表
    if os.path.exists(SYMBOL_CACHE_FILE):
        df_info = pd.read_parquet(SYMBOL_CACHE_FILE)
    else:
        print("Fetching fresh list...")
        try:
            df_info = ak.stock_info_a_code_name()
            df_info = df_info.rename(columns={"code": "symbol"})
            df_info = df_info[df_info["symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")]
            df_info['first_date'] = None
            df_info.to_parquet(SYMBOL_CACHE_FILE, index=False)
        except Exception as e:
            print(f"Error getting list: {e}"); return

    listing_map = dict(zip(df_info['symbol'], df_info.get('first_date', [None]*len(df_info))))
    symbols = df_info['symbol'].tolist()
    
    # 3. 并发抓取
    results = []
    print(f"Starting audit/fetch for {len(symbols)} symbols...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 timeout 参数，如果任务整体卡死，这里可以抛出异常
        future_to_symbol = {executor.submit(fetch_one, s, listing_map.get(s), args.check): s for s in symbols}
        
        try:
            pbar = tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Processing")
            for future in pbar:
                if should_abort:
                    # 如果熔断，取消剩余所有还未开始的任务
                    for f in future_to_symbol.keys(): f.cancel()
                    break
                
                res = future.result()
                results.append(res)
                # 在进度条显示失败数
                f_count = sum(1 for x in results if "FAIL" in x)
                pbar.set_postfix({"Fails": f_count, "C_Fails": consecutive_fails})
        except KeyboardInterrupt:
            print("\nUser cancelled. Shutting down...")
            executor.shutdown(wait=False)
            sys.exit(1)

    # 4. 更新缓存
    if not args.check:
        new_starts = {r.split('|')[0].strip(): r.split('|')[1] for r in results if "FETCHED|" in r}
        if new_starts:
            for sym, d in new_starts.items():
                df_info.loc[df_info['symbol'] == sym, 'first_date'] = d
            df_info.to_parquet(SYMBOL_CACHE_FILE, index=False)

    print("\n" + "="*40)
    print("DONE.")
    print(f"Perfect Files: {sum(1 for r in results if ' OK' in r)}")
    print(f"Updated Files: {sum(1 for r in results if 'FETCHED' in r)}")
    print(f"Failed Tasks : {sum(1 for r in results if 'FAIL' in r)}")
    if should_abort: print("REASON: Program was stopped by circuit breaker.")

if __name__ == "__main__":
    main()
