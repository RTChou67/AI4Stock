#!/usr/bin/env python3
"""
LightGBM 量化模型 - 训练脚本 (Sharpe Target)
配置：近5年数据(2020-2025)，1年滚动窗口
改进：使用 Risk-Adjusted Return (Sharpe) 作为训练目标
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import joblib
import contextlib
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "data/features"
MODEL_DIR = "models_lgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

N_JOBS = 16
MIN_PRICE, MAX_PRICE = 2.0, 1000.0
MIN_AVG_VOLUME = 1000000
MIN_LISTING_DAYS = 60

DATA_START_YEAR = 2020  
TRAINING_START_YEAR = 2021  
WEIGHT_HALF_LIFE = 60
GAP_DAYS = 30 

BASE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_60',
    'dist_ma5', 'dist_ma20', 'dist_ma60', 'dist_ma120',
    'std_20', 'std_60', 'atr_14',
    'vol_ratio_5', 'vol_ratio_20',
    'rsi_14', 'macd_diff', 'hl_range',
    'dist_high_20', 'dist_low_20'
]

# 调试计数器
DEBUG_FAIL_COUNT = 0
DEBUG_MAX_PRINT = 10

def filter_stock(df):
    if len(df) < MIN_LISTING_DAYS: return "Too few rows"
    if df['close'].isna().all() or df['volume'].isna().all(): return "All NaN"
    recent = df.tail(20)
    recent = recent[recent['close'] > 0]
    if len(recent) < 10: return "Recent < 10"
    avg_price = recent['close'].mean()
    if avg_price < MIN_PRICE or avg_price > MAX_PRICE: return f"Price {avg_price:.2f} out of range"
    avg_volume = recent['volume'].mean()
    if pd.isna(avg_volume) or avg_volume < MIN_AVG_VOLUME: return f"Vol {avg_volume:.0f} too low"
    return True


def process_one_file(filepath):
    global DEBUG_FAIL_COUNT
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 100: 
            if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Skip {filepath}: Raw len < 100")
            DEBUG_FAIL_COUNT += 1
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= f'{DATA_START_YEAR}-01-01']
        if len(df) < 100: 
            if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Skip {filepath}: Len after date filter < 100")
            DEBUG_FAIL_COUNT += 1
            return None
        
        filter_res = filter_stock(df)
        if filter_res is not True: 
            if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Skip {filepath}: {filter_res}")
            DEBUG_FAIL_COUNT += 1
            return None
        
        df = df.sort_values('date')
        
        if 'symbol' not in df.columns:
            symbol = os.path.splitext(os.path.basename(filepath))[0]
            df['symbol'] = symbol
            
        # ==== 加载 Sharpe Label ====
        if 'future_20d_sharpe' in df.columns:
            df['label'] = df['future_20d_sharpe']
        elif 'future_20d_ret' in df.columns:
            df['label'] = df['future_20d_ret']
        else:
            if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Skip {filepath}: No label col found")
            DEBUG_FAIL_COUNT += 1
            return None
            
        cols = ['date', 'symbol'] + BASE_FEATURES + ['label']
        missing = [c for c in cols if c not in df.columns]
        if missing: 
            if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Skip {filepath}: Missing features {missing}")
            DEBUG_FAIL_COUNT += 1
            return None
            
        return df[cols]
    except Exception as e:
        if DEBUG_FAIL_COUNT < DEBUG_MAX_PRINT: print(f"Error {filepath}: {e}")
        DEBUG_FAIL_COUNT += 1
        return None


def load_all_data(files):
    print(f"Loading feature data from {DATA_DIR}...")
    dfs = []
    for f in tqdm(files, desc="Loading"):
        result = process_one_file(f)
        if result is not None: dfs.append(result)
        
    print(f"Loaded {len(dfs)} valid stocks out of {len(files)}")
    if len(dfs) == 0: return None
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data.sort_values(['date', 'symbol'])


def add_cross_sectional_rank(all_data):
    # ==== 核心防污染机制：剔除妖股 ====
    print("Filtering extreme volatility outliers (Top 1% per day)...")
    # 计算每天 std_20 的百分位排名
    all_data['std_20_pct'] = all_data.groupby('date')['std_20'].transform(lambda x: x.rank(pct=True))
    # 丢弃每天波动率最大的前 1% 股票 (比如那些连板、天地板的妖股)
    # 这让模型不再被这些无法用正常逻辑预测的噪声数据带偏
    initial_len = len(all_data)
    all_data = all_data[all_data['std_20_pct'] <= 0.99].copy()
    all_data = all_data.drop(columns=['std_20_pct'])
    filtered_len = len(all_data)
    print(f"Dropped {initial_len - filtered_len} outlier rows.")

    print("Adding rank features...")
    rank_features = []
    for col in tqdm(BASE_FEATURES, desc="Ranking"):
        rank_col = f'{col}_rank'
        all_data[rank_col] = all_data.groupby('date')[col].transform(
            lambda x: x.rank(pct=True, method='average')
        )
        rank_features.append(rank_col)
    return all_data, rank_features


def compute_time_weights(dates, target_date, half_life=60):
    days_ago = (target_date - dates).dt.days
    days_ago = days_ago.clip(lower=0)
    weights = np.power(0.5, days_ago / half_life)
    weights = weights * len(weights) / weights.sum()
    return weights


def train_model_for_date(target_date, all_data, rank_features):
    date_str = target_date.strftime('%Y%m%d')
    model_path = os.path.join(MODEL_DIR, f"lgbm_{date_str}.pkl")
    
    available_data_end = target_date - pd.Timedelta(days=GAP_DAYS)
    
    valid_end = available_data_end
    valid_start = valid_end - pd.Timedelta(days=30)
    
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)
    
    train_mask = (all_data['date'] >= train_start) & (all_data['date'] <= train_end) & (all_data['label'].notna())
    valid_mask = (all_data['date'] >= valid_start) & (all_data['date'] <= valid_end) & (all_data['label'].notna())
    
    train_data = all_data[train_mask].copy()
    valid_data = all_data[valid_mask].copy()
    
    if len(train_data) < 5000 or len(valid_data) < 200:
        return f"{date_str}: Skipped (train={len(train_data)})"
    
    X_train = train_data[rank_features].values
    y_train = train_data['label'].values
    train_weights = compute_time_weights(train_data['date'], train_end, WEIGHT_HALF_LIFE)
    
    X_valid = valid_data[rank_features].values
    y_valid = valid_data['label'].values
    
    train_set = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    
    params = {
        'objective': 'regression', 
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': N_JOBS,
        'num_threads': N_JOBS
    }
    
    with contextlib.redirect_stdout(StringIO()):
        model = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[train_set, valid_set],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
    
    model_data = {
        'model': model,
        'features': rank_features,
        'best_iteration': model.best_iteration,
        'best_score': model.best_score,
        'train_date': target_date,
        'target_type': 'sharpe' 
    }
    joblib.dump(model_data, model_path)
    
    val_score = model.best_score.get('valid', {}).get('rmse', 0.0)
    return f"{date_str}: Done (Val RMSE: {val_score:.4f})"


def main():
    print("=" * 60)
    print("LightGBM Training (Sharpe Target)")
    print(f"Label: future_20d_sharpe (Future Ret / Future Vol)")
    print("=" * 60)
    
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    print(f"Found {len(files)} files.")
    
    print("\nStep 1: Loading data...")
    all_data = load_all_data(files)
    if all_data is None: 
        print("[ERROR] No valid data loaded! Check skip reasons above.")
        return
    
    print("\nStep 2: Adding rank features...")
    all_data, rank_features = add_cross_sectional_rank(all_data)
    
    print("\nStep 3: Training models...")
    all_dates = sorted(all_data['date'].unique())
    start_date = pd.Timestamp(f"{TRAINING_START_YEAR}-01-01")
    end_date = all_data['date'].max() - pd.Timedelta(days=5)
    
    train_dates = []
    current_date = start_date
    while current_date <= end_date:
        valid_dates = [d for d in all_dates if d >= current_date]
        if valid_dates: train_dates.append(valid_dates[0])
        current_date += pd.Timedelta(days=10)
    
    train_dates = sorted(list(set(train_dates)))
    print(f"Total models to train: {len(train_dates)}")
    
    for target_date in tqdm(train_dates, desc="Training"):
        try:
            date_str = target_date.strftime('%Y%m%d')
            model_path = os.path.join(MODEL_DIR, f"lgbm_{date_str}.pkl")
            if os.path.exists(model_path): continue
            train_model_for_date(target_date, all_data, rank_features)
        except Exception as e:
            print(f"Error {target_date}: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
