import akshare as ak
import akshare_proxy_patch
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
AUTH_TOKEN = os.getenv("AKSHARE_PROXY_TOKEN", "")
if AUTH_TOKEN:
    akshare_proxy_patch.install_patch("101.201.173.125", AUTH_TOKEN, 3)

df = ak.stock_zh_a_spot_em()
df = df.rename(columns={'代码': 'symbol'})

print(f"Total rows from EastMoney spot: {len(df)}")

# 提取所有前缀(前3位)
df['prefix'] = df['symbol'].str[:3]
counts = df['prefix'].value_counts()

print("\nPrefix distribution:")
for prefix, count in counts.items():
    print(f"  {prefix}: {count}")

# 测试我们的正则表达式
import re
regex = r"^(000|001|002|003|300|301|600|601|603|605|688)"
matched = df[df['symbol'].str.match(regex)]
print(f"\nMatched by our regex: {len(matched)}")

# 查看以 00, 30, 60, 68 开头但没被正则匹配到的
broad_regex = r"^(00|30|60|68)"
broad_matched = df[df['symbol'].str.match(broad_regex)]
missed = broad_matched[~broad_matched['symbol'].str.match(regex)]
if len(missed) > 0:
    print(f"\nMissed by strict regex but broadly fit: {len(missed)}")
    print(missed['prefix'].value_counts())
