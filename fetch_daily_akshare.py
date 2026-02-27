import os
import time
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ======================
# 配置
# ======================
DATA_DIR = "data/daily"
START_DATE = "20000101"
END_DATE = "20251231"

MAX_WORKERS = 8      # 并发线程数（建议 6~12）
RETRY = 3
SLEEP = 0.1          # 每个请求后休眠

os.makedirs(DATA_DIR, exist_ok=True)


# ======================
# 获取A股列表
# ======================
def get_a_share_list():
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "symbol"})

    df = df[df["symbol"].str.match(
        r"^(000|001|002|003|300|301|600|601|603|605|688)"
    )]

    return df["symbol"].tolist()


# ======================
# 单只股票抓取
# ======================
def fetch_one(symbol):

    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")

    # ===== 全量更新 (避免复权因子不一致) =====
    # qfq 数据每次抓取都会根据当前复权因子调整历史价格
    # 因此不能简单 append，必须全量覆盖
    start_date = START_DATE
    
    # 简单的跳过逻辑：如果文件是今天生成的，跳过
    if os.path.exists(file_path):
        try:
            mtime = os.path.getmtime(file_path)
            if pd.Timestamp.fromtimestamp(mtime).date() == pd.Timestamp.today().date():
                return f"{symbol} skip (fresh)"
        except:
            pass

    # ===== 重试 =====
    for attempt in range(RETRY):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=END_DATE,
                adjust="qfq",
            )

            if df is None or df.empty:
                return f"{symbol} empty"

            df = df.rename(columns={
                "日期": "date",
                "股票代码": "symbol",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
                "涨跌幅": "pct_change",
            })

            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            df = df.sort_values("date")

            # 直接保存（覆盖旧文件）
            df.to_parquet(file_path, index=False)

            time.sleep(SLEEP)
            return f"{symbol} ok"

        except Exception as e:
            time.sleep(1 + attempt)

    return f"{symbol} fail"


# ======================
# 主流程（并发）
# ======================
def main():
    symbols = get_a_share_list()
    print(f"Total symbols: {len(symbols)}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    # 统计结果
    ok = sum("ok" in r for r in results)
    fail = sum("fail" in r for r in results)
    skip = sum("skip" in r for r in results)

    print(f"OK: {ok}, FAIL: {fail}, SKIP: {skip}")


if __name__ == "__main__":
    main()
