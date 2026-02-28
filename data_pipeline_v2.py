#!/usr/bin/env python3
"""
AI4Stock 数据管线 V2 (Cookie Authentication Edition)
思路：通过劫持 requests 模块，注入真实浏览器 Cookie 和仿真 Header 绕过反爬。
用法：
    1. 使用 EditThisCookie 导出东财 JSON 格式 Cookie 到 data/cookies.json
    2. python data_pipeline_v2.py --fetch --hist
"""

import os
import time
import json
import pandas as pd
import akshare as ak
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys
import threading
import pyarrow.parquet as pq
import datetime

# ======================
# 全局配置
# ======================
RAW_DAILY_DIR = "data/raw/daily"
RAW_VAL_DIR = "data/raw/valuation"
PROCESSED_DIR = "data/processed/combined"
CACHE_DIR = "data"
SYMBOL_CACHE_FILE = os.path.join(CACHE_DIR, "symbols_cache.parquet")
COOKIES_FILE = os.path.join(CACHE_DIR, "cookies.json")

# 自动获取今天日期
_now = datetime.datetime.now()
AK_START_STR = "19900101"
AK_END_STR = _now.strftime("%Y%m%d")

# 初始基准（默认为 7 天前，稍后在 main 中通过探测更新为真实的最新交易日）
TARGET_END_DATE = pd.Timestamp((_now - datetime.timedelta(days=7)).strftime("%Y-%m-%d"))

import random

def update_target_end_date(args):
    """通过探测平安银行的数据，确定全市场真实的最新交易日"""
    global TARGET_END_DATE
    print("[*] Probing latest trading date...")
    try:
        # 探测最近 20 天的数据
        start_probe = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime("%Y%m%d")
        # 模拟真实的行情页访问
        headers = DEFAULT_HEADERS.copy()
        headers["Referer"] = "https://quote.eastmoney.com/sz000001.html"
        
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=start_probe, end_date=AK_END_STR, adjust="hfq")
        if not df.empty:
            real_latest = pd.Timestamp(df['日期'].max())
            TARGET_END_DATE = real_latest
            print(f"[+] Latest market trading date detected: {TARGET_END_DATE.date()}")
            return True
        else:
            print("[!] Probe returned empty data. Using fallback date.")
    except Exception as e:
        print(f"\n[!] Network probe failed: {e}")
        print("[!] Using safety fallback: {}. Connection might be throttled.".format(TARGET_END_DATE.date()))
    return False # 不再强制 sys.exit，允许尝试后续任务

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/center/gridlist.html",
    "Connection": "keep-alive"
}

DAILY_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
VAL_RENAME_MAP = {
    '数据日期': 'date', '当日收盘价': 'v_close', '总市值': 'total_mv', '流通市值': 'circ_mv',
    '总股本': 'total_share', '流通股本': 'circ_share', 'PE(TTM)': 'pe_ttm', 'PE(静)': 'pe_static',
    '市净率': 'pb', 'PEG值': 'peg', '市现率': 'pcf', '市销率': 'ps'
}

# 全局状态
fail_lock = threading.Lock()
con_fails = 0
should_abort = False

for d in [RAW_DAILY_DIR, RAW_VAL_DIR, PROCESSED_DIR]: os.makedirs(d, exist_ok=True)

# ======================
# Cookie 注入与 Requests 劫持 (升级版 - 模拟 TLS 指纹)
# ======================
from curl_cffi import requests as curleq

class RequestPatcher:
    def __init__(self):
        # 使用 curl_cffi 的 Session，它可以模拟 Chrome 的 TLS 握手特征
        self.session = curleq.Session(impersonate="chrome120")
        self.session.headers.update(DEFAULT_HEADERS)

    def load_cookies(self):
        if not os.path.exists(COOKIES_FILE):
            print(f"[!] Warning: {COOKIES_FILE} not found. Running without custom cookies.")
            return False
        try:
            with open(COOKIES_FILE, 'r') as f:
                cookie_list = json.load(f)
            # 建立一个普通的 cookie 字典传给 curl_cffi
            cookies = {c['name']: c['value'] for c in cookie_list}
            self.session.cookies.update(cookies)
            print(f"[*] Successfully loaded {len(cookie_list)} cookies from {COOKIES_FILE}")
            return True
        except Exception as e:
            print(f"[!] Error loading cookies: {e}")
            return False

    def verify(self):
        try:
            # 访问东财首页验证，模拟 Chrome
            resp = self.session.get("https://www.eastmoney.com/", timeout=10)
            if resp.status_code == 200 and "拒绝访问" not in resp.text:
                print("[+] Browser Impersonation & Cookies Verification PASSED.")
                return True
            else:
                print(f"[-] Verification FAILED. Status: {resp.status_code}")
                return False
        except Exception as e:
            print(f"[-] Verification ERROR: {e}")
            return False

    def patch(self):
        # 劫持全局 requests 模块，将所有请求重定向到我们的 curl_cffi session
        def patched_get(url, **kwargs):
            # 清理 requests 特有的参数，避免传给 curl_cffi 报错
            kwargs.pop('session', None)
            kwargs.pop('verify', None)
            if 'timeout' not in kwargs: kwargs['timeout'] = 30
            # 统一将 stream 改为非流式，防止 akshare 内部逻辑冲突
            kwargs.pop('stream', None)
            
            return self.session.get(url, **kwargs)

        def patched_post(url, **kwargs):
            kwargs.pop('session', None)
            kwargs.pop('verify', None)
            kwargs.pop('stream', None)
            if 'timeout' not in kwargs: kwargs['timeout'] = 30
            return self.session.post(url, **kwargs)

        # 核心劫持
        import requests
        requests.get = patched_get
        requests.post = patched_post
        print("[*] Global Requests Hijacked with curl_cffi (Chrome Impersonation).")

# ======================
# 核心逻辑
# ======================
def check_status(symbol):
    def check_one(path, target_cols, date_col='date'):
        if not os.path.exists(path): return False
        try:
            meta = pq.read_metadata(path)
            if meta.num_rows == 0: return False
            if not all(c in meta.schema.names for c in target_cols): return False
            col_idx = meta.schema.names.index(date_col)
            rg = meta.row_group(meta.num_row_groups-1)
            col_meta = rg.column(col_idx)
            if col_meta.is_stats_set:
                return pd.Timestamp(col_meta.statistics.max) >= TARGET_END_DATE
            return False
        except: return False
    d_ok = check_one(os.path.join(RAW_DAILY_DIR, f"{symbol}.parquet"), DAILY_COLS)
    v_ok = check_one(os.path.join(RAW_VAL_DIR, f"{symbol}.parquet"), ['数据日期', 'PE(TTM)'], '数据日期')
    return d_ok, v_ok

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

def fetch_and_fuse(symbol, args):
    global con_fails, should_abort
    if should_abort: return f"{symbol} ABORTED"
    
    d_ok, v_ok = check_status(symbol)
    
    # 检查 Processed 文件是否存在
    processed_path = os.path.join(PROCESSED_DIR, f"{symbol}.parquet")
    p_ok = os.path.exists(processed_path)

    # --no-update 逻辑
    if args.no_update:
        d_exists = os.path.exists(os.path.join(RAW_DAILY_DIR, f"{symbol}.parquet"))
        v_exists = os.path.exists(os.path.join(RAW_VAL_DIR, f"{symbol}.parquet"))
        if d_exists and v_exists and p_ok:
            with fail_lock: con_fails = 0
            return f"{symbol} OK"
        if d_exists: d_ok = True
        if v_exists: v_ok = True

    if d_ok and v_ok and p_ok and not args.fuse:
        with fail_lock: con_fails = 0
        return f"{symbol} OK"
    if args.check: return f"{symbol} INCOMPLETE"

    # 1. 抓取行情
    file_path_d = os.path.join(RAW_DAILY_DIR, f"{symbol}.parquet")
    current_start_d = AK_START_STR
    existing_df_d = None
    
    if os.path.exists(file_path_d):
        try:
            existing_df_d = pd.read_parquet(file_path_d)
            if not existing_df_d.empty:
                last_d = pd.to_datetime(existing_df_d['date']).max()
                if last_d < TARGET_END_DATE:
                    current_start_d = (last_d + pd.Timedelta(days=1)).strftime("%Y%m%d")
                else:
                    current_start_d = None
        except: pass

    if current_start_d:
        try:
            if args.sina:
                sina_sym = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
                df_new = ak.stock_zh_a_daily(symbol=sina_sym, start_date=current_start_d, end_date=AK_END_STR, adjust="hfq")
                if df_new is not None and not df_new.empty:
                    df_new = df_new.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "amount": "amount", "turnover": "turnover"})
                    df_new['volume'] = df_new['volume'] / 100.0
                    df_new['turnover'] = df_new['turnover'] * 100.0
            else:
                df_new = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=current_start_d, end_date=AK_END_STR, adjust="hfq")
                if df_new is not None and not df_new.empty:
                    df_new = df_new.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"})
            
            if df_new is not None and not df_new.empty:
                df_new["date"] = pd.to_datetime(df_new["date"])
                df_new["symbol"] = symbol
                if existing_df_d is not None:
                    df_d = pd.concat([existing_df_d, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
                else:
                    df_d = df_new.sort_values('date')
                # 使用优化后的保存函数
                save_optimized_parquet(df_d[DAILY_COLS], file_path_d)
            elif existing_df_d is None:
                raise ValueError("No data returned")
        except Exception as e:
            with fail_lock: con_fails += 1
            if con_fails >= 10: should_abort = True
            return f"{symbol} DAILY_FAIL ({str(e)[:20]})"

    # 2. 抓取估值
    file_path_v = os.path.join(RAW_VAL_DIR, f"{symbol}.parquet")
    if not v_ok:
        try:
            df_v = ak.stock_value_em(symbol=symbol)
            if df_v is not None and not df_v.empty:
                save_optimized_parquet(df_v, file_path_v)
            else: raise ValueError("Empty VAL")
        except Exception as e: return f"{symbol} VAL_FAIL ({str(e)[:20]})"

    # 3. 融合
    try:
        df_d = pd.read_parquet(file_path_d)
        df_v = pd.read_parquet(file_path_v).rename(columns=VAL_RENAME_MAP)
        df_d['date'], df_v['date'] = pd.to_datetime(df_d['date']), pd.to_datetime(df_v['date'])
        df = pd.merge(df_d, df_v, on='date', how='outer').sort_values('date')
        if 'v_close' in df.columns:
            df['close'] = df['close'].fillna(df['v_close'])
            df = df.drop(columns=['v_close'])
        for c in ['open', 'high', 'low']: 
            if c in df.columns: df[c] = df[c].fillna(df['close'])
        df['symbol'] = symbol
        save_optimized_parquet(df, processed_path)
        with fail_lock: con_fails = 0
        time.sleep(args.sleep)
        actual_start = df['date'].min()
        return f"{symbol} FUSED|{actual_start.strftime('%Y-%m-%d')}"
    except Exception as e: return f"{symbol} FUSE_ERR ({str(e)[:20]})"

# ======================
# 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--no-update", action="store_true", help="Don't update existing files, only fetch missing ones")
    parser.add_argument("--sina", action="store_true")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.check:
        args.workers, args.sleep = 16, 0.0
    else:
        # 初始化劫持
        patcher = RequestPatcher()
        patcher.load_cookies()
        if not patcher.verify():
            print("[!] Warning: Initial verification failed. Continuing anyway...")
        patcher.patch()
        
        # 探测最新交易日
        update_target_end_date(args)
        
        if args.workers is None: args.workers = 4 # Cookie 模式建议中等并发

    if os.path.exists(SYMBOL_CACHE_FILE):
        symbols = pd.read_parquet(SYMBOL_CACHE_FILE)['symbol'].tolist()
    else:
        print("No symbol cache. Please run data_pipeline.py once."); return

    print(f"\nMode: {'CHECK' if args.check else 'V2 (COOKIE PATCH)'} | Workers: {args.workers}")
    
    results = []
    fail_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {executor.submit(fetch_and_fuse, s, args): s for s in symbols}
        try:
            # 增加 mininterval 确保高频更新，移除 O(N) 的 Fails 统计
            pbar = tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Process", mininterval=0.1)
            for future in pbar:
                if should_abort:
                    for f in future_to_symbol.keys(): f.cancel()
                    print("\n[!] Circuit breaker active. Force shutting down...")
                    break
                
                res = future.result()
                results.append(res)
                
                if "FAIL" in res or "ERR" in res:
                    fail_count += 1
                
                pbar.set_postfix({"Fails": fail_count, "Consec": con_fails})
        except KeyboardInterrupt:
            print("\n[!] User interrupted. Cleaning up...")
            for f in future_to_symbol.keys(): f.cancel()
            executor.shutdown(wait=False)
            sys.exit(1)
        
        # 如果熔断，不等待已经挂起的线程，直接关掉（虽然 Python 线程无法真正被 Kill，但我们可以不 Wait）
        if should_abort:
            executor.shutdown(wait=False)
            # 在这里直接进入后续的 log 写入逻辑，不被挂起的线程卡死

    # 写入失败日志
    fail_log_path = "fetch_fails_v2.log"
    fails_and_errors = [r for r in results if any(x in r for x in ["FAIL", "ERR", "INCOMPLETE", "MISSING"])]
    if fails_and_errors:
        with open(fail_log_path, "w", encoding="utf-8") as f:
            f.write(f"--- V2 Pipeline Report: {pd.Timestamp.now()} ---\n")
            for r in fails_and_errors:
                f.write(f"{r}\n")
        print(f"\n[INFO] Detailed failures saved to {fail_log_path}")

    print("\n" + "="*40)
    print("DONE.")
    print(f"Perfect Files: {sum(1 for r in results if ' OK' in r)}")

    if args.check:
        incomplete_count = sum(1 for r in results if "INCOMPLETE" in r)
        print(f"Incomplete Files: {incomplete_count}")
        if incomplete_count > 0:
            print(f"[ACTION] Run with --fetch to fix {incomplete_count} files.")
    else:
        print(f"Updated Files: {sum(1 for r in results if 'FUSED' in r)}")
        print(f"Failed Tasks : {sum(1 for r in results if 'FAIL' in r or 'ERR' in r)}")

if __name__ == "__main__":
    main()
