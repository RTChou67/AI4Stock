#!/usr/bin/env python3
"""
统一 A 股日线数据抓取脚本 (支持东财 hist 和新浪 daily 接口)
包含：完整性检查、增量/全量更新、代理支持、断点熔断
"""
import os
import time
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys
import threading
from dotenv import load_dotenv

# ======================
# 全局配置
# ======================
DATA_DIR = "data/daily"
CACHE_DIR = "data"
SYMBOL_CACHE_FILE = os.path.join(CACHE_DIR, "symbols_cache.parquet")

TARGET_START_DATE = pd.Timestamp("2021-01-04")
TARGET_END_DATE = pd.Timestamp("2025-12-31")

AK_START_DATE_STR = "20210101"
AK_END_DATE_STR = "20251231"

EXPECTED_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']

# 并发与熔断控制
MAX_WORKERS = 8 # 默认并发，根据源和网络调整
SLEEP = 0.5     # 默认休眠，根据源调整
CIRCUIT_BREAKER_LIMIT = 30 

fail_lock = threading.Lock()
consecutive_fails = 0
should_abort = False

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ======================
# 初始化代理 (仅对东财生效)
# ======================
def init_proxy():
    load_dotenv()
    AUTH_TOKEN = os.getenv("AKSHARE_PROXY_TOKEN", "")
    if AUTH_TOKEN:
        try:
            import akshare_proxy_patch
            # 重试次数设为 3，避免单次请求挂起太久
            akshare_proxy_patch.install_patch("101.201.173.125", AUTH_TOKEN, 3)
            print("Proxy patch installed.")
        except ImportError:
            print("Warning: akshare-proxy-patch not installed. Running without proxy.")
    else:
        print("Warning: AKSHARE_PROXY_TOKEN not found in .env. Running without proxy.")


# ======================
# 辅助函数：新浪代码转换
# ======================
def format_symbol_for_sina(symbol):
    if symbol.startswith('6'): return f"sh{symbol}"
    if symbol.startswith('0') or symbol.startswith('3'): return f"sz{symbol}"
    if symbol.startswith('4') or symbol.startswith('8'): return f"bj{symbol}"
    return symbol


# ======================
# 核心验证逻辑
# ======================
def is_file_qualified(symbol, known_start_date=None):
    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if not os.path.exists(file_path): return False, "Missing"
    try:
        df = pd.read_parquet(file_path)
        if df.empty: return False, "Empty"
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 检查列
        if not all(col in df.columns for col in EXPECTED_COLS):
            return False, "Missing columns"
            
        start_date, end_date = df['date'].min(), df['date'].max()
        
        if end_date < TARGET_END_DATE: 
            return False, f"End date early: {end_date.date()}"
            
        effective_target = TARGET_START_DATE
        if known_start_date is not None: 
            effective_target = max(TARGET_START_DATE, pd.Timestamp(known_start_date))
            
        if start_date > (effective_target + pd.Timedelta(days=3)): 
            return False, f"Start date late: {start_date.date()}"
            
        return True, "Perfect"
    except: 
        return False, "Corrupted"


# ======================
# 抓取逻辑 (聚合版)
# ======================
def fetch_one(symbol, known_start, args):
    global consecutive_fails, should_abort
    
    if should_abort: return f"{symbol} ABORTED"

    qualified, reason = is_file_qualified(symbol, known_start)
    if qualified:
        with fail_lock: consecutive_fails = 0
        return f"{symbol} OK"
        
    if args.check: return f"{symbol} INCOMPLETE ({reason})"

    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    
    try:
        if args.sina:
            # === 新浪源 (stock_zh_a_daily) ===
            sina_symbol = format_symbol_for_sina(symbol)
            df = ak.stock_zh_a_daily(
                symbol=sina_symbol, 
                start_date=AK_START_DATE_STR, 
                end_date=AK_END_DATE_STR,
                adjust="qfq"
            )
            if df is None or df.empty: raise ValueError("Empty Sina API")
            
            # 适配新浪列名
            rename_map = {
                "日期": "date", "date": "date", "开盘": "open", "open": "open",
                "收盘": "close", "close": "close", "最高": "high", "high": "high",
                "最低": "low", "low": "low", "成交量": "volume", "volume": "volume",
                "成交额": "amount", "amount": "amount", "换手率": "turnover", "turnover": "turnover",
                "vol": "volume"
            }
            df = df.rename(columns=rename_map)
            
            # 统一单位：新浪 volume 是股，转为手 (与 hist 一致)
            if 'volume' in df.columns:
                df['volume'] = df['volume'] / 100.0
                
            # 统一单位：新浪 turnover 是小数，转为百分比 (与 hist 一致)
            if 'turnover' in df.columns:
                df['turnover'] = df['turnover'] * 100.0

        else:
            # === 东财源 (stock_zh_a_hist) ===
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=AK_START_DATE_STR, end_date=AK_END_DATE_STR,
                adjust="qfq"
            )
            if df is None or df.empty: raise ValueError("Empty EastMoney API")
            
            df = df.rename(columns={
                "日期": "date", "股票代码": "symbol", "开盘": "open", "收盘": "close", 
                "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"
            })

        # === 共同处理逻辑 ===
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = str(symbol)
        
        # 确保包含所有必需列 (容错)
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0.0 if col != 'date' and col != 'symbol' else None

        df = df[EXPECTED_COLS]
        df = df[df['date'] >= TARGET_START_DATE]
        
        if df.empty:
            raise ValueError("No data after 2021")
            
        actual_start = df['date'].min()
        df.sort_values("date").to_parquet(file_path, index=False)
        
        with fail_lock: consecutive_fails = 0
        time.sleep(args.sleep)
        return f"{symbol} FETCHED|{actual_start.strftime('%Y-%m-%d')}"

    except Exception as e:
        with fail_lock:
            consecutive_fails += 1
            if consecutive_fails >= CIRCUIT_BREAKER_LIMIT:
                should_abort = True
                print(f"\n[!!!] CIRCUIT BREAKER: {CIRCUIT_BREAKER_LIMIT} consecutive fails.")
        
        time.sleep(args.sleep)
        return f"{symbol} FAIL ({str(e)[:30]})"


# ======================
# 获取列表
# ======================
def get_list(args):
    if os.path.exists(SYMBOL_CACHE_FILE):
        return pd.read_parquet(SYMBOL_CACHE_FILE)
    
    if args.check:
        print(f"Error: {SYMBOL_CACHE_FILE} not found. Cannot run check without local cache.")
        return None
        
    print("Fetching fresh symbol list...")
    try:
        df_info = ak.stock_info_a_code_name()
        df_info = df_info.rename(columns={"code": "symbol"})
        df_info = df_info[df_info["symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")]
        df_info['first_date'] = None
        df_info.to_parquet(SYMBOL_CACHE_FILE, index=False)
        return df_info
    except Exception as e:
        print(f"Error getting list: {e}")
        return None


# ======================
# 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only check integrity")
    parser.add_argument("--sina", action="store_true", help="Use Sina daily API (fast but unstable IP limit)")
    parser.add_argument("--hist", action="store_true", help="Use EastMoney hist API (high quality, needs proxy)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of concurrent workers")
    parser.add_argument("--sleep", type=float, default=SLEEP, help="Sleep time between requests")
    args = parser.parse_args()

    if not args.check and not args.sina and not args.hist:
        print("Please specify a data source: --hist OR --sina")
        print("Or use --check to audit local data.")
        return

    # Check 模式极速配置
    if args.check:
        args.workers = 16
        args.sleep = 0.0

    # 安全起见，如果用东财且没开 proxy，强烈建议单线程长休眠
    if args.hist and not args.check:
        init_proxy()
        # 自动覆盖保守参数 (用户没手动指定的话)
        if args.workers == MAX_WORKERS: args.workers = 1
        if args.sleep == SLEEP: args.sleep = 1.0
    
    # 新浪接口通常可以多开
    if args.sina and not args.check:
        if args.workers == MAX_WORKERS: args.workers = 4
        if args.sleep == SLEEP: args.sleep = 0.2

    df_info = get_list(args)
    if df_info is None or df_info.empty: return

    listing_map = dict(zip(df_info['symbol'], df_info.get('first_date', [None]*len(df_info))))
    symbols = df_info['symbol'].tolist()
    
    print(f"\nMode: {'CHECK ONLY' if args.check else ('SINA (daily)' if args.sina else 'EASTMONEY (hist)')}")
    print(f"Targets: {len(symbols)} | Workers: {args.workers} | Sleep: {args.sleep}s")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {executor.submit(fetch_one, s, listing_map.get(s), args): s for s in symbols}
        
        try:
            pbar = tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Processing")
            for future in pbar:
                if should_abort:
                    for f in future_to_symbol.keys(): f.cancel()
                    break
                res = future.result()
                results.append(res)
                f_count = sum(1 for x in results if "FAIL" in x)
                pbar.set_postfix({"Fails": f_count, "Consec": consecutive_fails})
        except KeyboardInterrupt:
            print("\nUser cancelled. Shutting down...")
            executor.shutdown(wait=False)
            sys.exit(1)

    # 更新缓存 (仅更新新获取成功的部分)
    if not args.check:
        new_starts = {r.split('|')[0].strip().split()[0]: r.split('|')[1] for r in results if "FETCHED|" in r}
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
