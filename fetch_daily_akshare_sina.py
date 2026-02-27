import os
import time
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ======================
# 配置：新浪源
# ======================
DATA_DIR = "data/daily"
START_DATE = "20000101"
END_DATE = "20251231"

MAX_WORKERS = 8      # 新浪接口通常允许更高并发
RETRY = 3
SLEEP = 0.2

os.makedirs(DATA_DIR, exist_ok=True)

def get_a_share_list():
    try:
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "symbol"})
        df = df[df["symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")]
        return df["symbol"].tolist()
    except Exception as e:
        print(f"Error getting stock list: {e}")
        return []

def format_symbol_for_sina(symbol):
    if symbol.startswith('6'): return f"sh{symbol}"
    elif symbol.startswith('0') or symbol.startswith('3'): return f"sz{symbol}"
    elif symbol.startswith('4') or symbol.startswith('8'): return f"bj{symbol}"
    return symbol

def fetch_one(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    
    if os.path.exists(file_path):
        try:
            mtime = os.path.getmtime(file_path)
            if pd.Timestamp.fromtimestamp(mtime).date() == pd.Timestamp.today().date():
                return f"{symbol} skip"
        except:
            pass

    sina_symbol = format_symbol_for_sina(symbol)
    
    for attempt in range(RETRY):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="qfq"
            )

            if df is None or df.empty:
                return f"{symbol} empty"

            rename_map = {
                "日期": "date", "date": "date",
                "开盘": "open", "open": "open",
                "收盘": "close", "close": "close",
                "最高": "high", "high": "high",
                "最低": "low", "low": "low",
                "成交量": "volume", "volume": "volume",
                "成交额": "amount", "amount": "amount",
                "换手率": "turnover", "turnover": "turnover"
            }
            df = df.rename(columns=rename_map)
            
            # 兼容列名
            if 'vol' in df.columns and 'volume' not in df.columns:
                df = df.rename(columns={'vol': 'volume'})

            # 校验
            if 'close' not in df.columns:
                return f"{symbol} bad_cols"

            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            df = df.sort_values("date")
            
            if 'amount' not in df.columns:
                df['amount'] = df['volume'] * df['close'] 
            if 'turnover' not in df.columns:
                df['turnover'] = 0.0

            df.to_parquet(file_path, index=False)
            time.sleep(SLEEP)
            return f"{symbol} ok"

        except Exception as e:
            time.sleep(1 + attempt)
    
    return f"{symbol} fail"

def main():
    print("Fetching with ak.stock_zh_a_daily (Sina)...")
    symbols = get_a_share_list()
    if not symbols: return

    print(f"Total symbols: {len(symbols)}")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    ok = sum("ok" in r for r in results)
    fail = sum("fail" in r for r in results)
    skip = sum("skip" in r for r in results)
    print(f"OK: {ok}, FAIL: {fail}, SKIP: {skip}")

if __name__ == "__main__":
    main()
