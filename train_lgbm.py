#!/usr/bin/env python3
"""
LightGBM 量化模型 - 训练脚本 (Fixed Leakage)
配置：近5年数据(2020-2025)，1年滚动窗口，时间衰减权重，预测未来20日
修复：增加 GAP_DAYS 确保训练时不使用未来数据作为 Label
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

# 忽略 pandas 的某些警告
warnings.filterwarnings('ignore')

DATA_DIR = "data/daily"
MODEL_DIR = "models_lgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

N_JOBS = 16
MIN_PRICE, MAX_PRICE = 2.0, 1000.0
MIN_AVG_VOLUME = 1000000
MIN_LISTING_DAYS = 60

# 数据时间范围配置
DATA_START_YEAR = 2020  # 加载数据的起始年份
TRAINING_START_YEAR = 2021  # 开始训练模型的年份

# 时间衰减权重参数
WEIGHT_HALF_LIFE = 60

# 关键修复：防止数据泄漏的间隔天数
# 预测 future_ret_20 (20个交易日) 约等于 30 个自然日
# 我们必须确保训练集的最后一天至少在 target_date 之前的 30 天，这样它的 label (未来20天收益) 才是已知的
GAP_DAYS = 30 

BASE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_40', 'ret_60',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio',
    'volatility_5', 'volatility_10', 'volatility_20',
    'inv_volatility_5', 'inv_volatility_10', 'inv_volatility_20',
    'vol_ratio', 'vol_ratio_20', 'new_high_ratio',
]


def filter_stock(df):
    if len(df) < MIN_LISTING_DAYS:
        return False
    if df['close'].isna().all() or df['volume'].isna().all():
        return False
    recent = df.tail(20)
    recent = recent[recent['close'] > 0]
    if len(recent) < 10:
        return False
    avg_price = recent['close'].mean()
    if avg_price < MIN_PRICE or avg_price > MAX_PRICE:
        return False
    avg_volume = recent['volume'].mean()
    if pd.isna(avg_volume) or avg_volume < MIN_AVG_VOLUME:
        return False
    return True


def compute_features(df):
    df = df.copy()
    df = df[df['close'] > 0]
    df = df[df['volume'] > 0]
    
    df['ret_1'] = df['close'].pct_change(1).clip(-0.2, 0.2)
    df['ret_5'] = df['close'].pct_change(5).clip(-0.5, 0.5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20).clip(-1, 1)
    df['ret_40'] = df['close'].pct_change(40)
    df['ret_60'] = df['close'].pct_change(60)
    
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma5_ratio'] = (df['close'] / df['ma5']).clip(0.5, 2)
    df['ma10_ratio'] = (df['close'] / df['ma10']).clip(0.5, 2)
    df['ma20_ratio'] = (df['close'] / df['ma20']).clip(0.5, 2)
    
    df['volatility_5'] = df['ret_1'].rolling(5).std().clip(lower=1e-4)
    df['volatility_10'] = df['ret_1'].rolling(10).std().clip(lower=1e-4)
    df['volatility_20'] = df['ret_1'].rolling(20).std().clip(lower=1e-4)
    df['inv_volatility_5'] = 1 / df['volatility_5']
    df['inv_volatility_10'] = 1 / df['volatility_10']
    df['inv_volatility_20'] = 1 / df['volatility_20']
    
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    df['vol_ratio_20'] = df['volume'] / df['vol_ma20']
    
    df['high_250'] = df['close'].rolling(250).max()
    df['new_high_ratio'] = (df['close'] / df['high_250']).clip(0, 2)
    
    # Label: 未来20日收益 (Shift -20)
    # 注意：Shift(-20) 意味着第 T 天的 label 需要第 T+20 天的数据
    df['future_ret_20'] = (df['close'].shift(-20) / df['close'] - 1).clip(-0.3, 0.3)
    
    return df


def process_one_file(filepath):
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 100:
            return None
        
        # 提前过滤：只加载2020年后的数据
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= f'{DATA_START_YEAR}-01-01']
        if len(df) < 100:
            return None
        
        if not filter_stock(df):
            return None
        
        df = df.sort_values('date')
        df = compute_features(df)
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        df['symbol'] = symbol
        # 不在这里 dropna future_ret_20，因为预测时需要最新数据
        # 训练时会自动 dropna
        cols = ['date', 'symbol'] + BASE_FEATURES + ['future_ret_20']
        df = df[cols] 
        return df if len(df) >= 60 else None
    except:
        return None


def load_all_data(files):
    print(f"Loading data from {DATA_START_YEAR} onwards...")
    dfs = []
    for f in tqdm(files, desc="Loading"):
        result = process_one_file(f)
        if result is not None:
            dfs.append(result)
    if len(dfs) == 0:
        return None
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data.sort_values(['date', 'symbol'])


def add_cross_sectional_rank(all_data):
    print("Adding rank features...")
    rank_features = []
    for col in tqdm(BASE_FEATURES, desc="Ranking"):
        rank_col = f'{col}_rank'
        # 使用 groupby transform 进行截面排名
        all_data[rank_col] = all_data.groupby('date')[col].transform(
            lambda x: x.rank(pct=True, method='average')
        )
        rank_features.append(rank_col)
    return all_data, rank_features


def compute_time_weights(dates, target_date, half_life=60):
    """计算时间衰减权重"""
    days_ago = (target_date - dates).dt.days
    days_ago = days_ago.clip(lower=0)
    weights = np.power(0.5, days_ago / half_life)
    # 归一化权重
    weights = weights * len(weights) / weights.sum()
    return weights


def train_model_for_date(target_date, all_data, rank_features):
    """为特定日期训练模型，确保不使用未来数据"""
    date_str = target_date.strftime('%Y%m%d')
    model_path = os.path.join(MODEL_DIR, f"lgbm_{date_str}.pkl")
    
    # ==== 关键修复：时间窗口划分 ====
    # 假设 target_date 是 "今天"
    # 我们只能使用 label 已经确定的数据。
    # Label 是未来20天收益，所以必须回退 GAP_DAYS (30天)
    
    available_data_end = target_date - pd.Timedelta(days=GAP_DAYS)
    
    # 验证集：可用数据的最后 30 天
    valid_end = available_data_end
    valid_start = valid_end - pd.Timedelta(days=30)
    
    # 训练集：验证集之前的 365 天
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)
    
    # 筛选数据 (必须 dropna，确保 label 存在)
    train_mask = (all_data['date'] >= train_start) & (all_data['date'] <= train_end) & (all_data['future_ret_20'].notna())
    valid_mask = (all_data['date'] >= valid_start) & (all_data['date'] <= valid_end) & (all_data['future_ret_20'].notna())
    
    train_data = all_data[train_mask].copy()
    valid_data = all_data[valid_mask].copy()
    
    if len(train_data) < 5000 or len(valid_data) < 200:
        return f"{date_str}: Skipped (train={len(train_data)}, valid={len(valid_data)})"
    
    X_train = train_data[rank_features].values
    y_train = train_data['future_ret_20'].values
    # 权重计算基准日期设为 train_end，越接近训练结束权重越高
    train_weights = compute_time_weights(train_data['date'], train_end, WEIGHT_HALF_LIFE)
    
    X_valid = valid_data[rank_features].values
    y_valid = valid_data['future_ret_20'].values
    
    train_set = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    
    params = {
        'objective': 'regression_l1', # 预测收益率，用L1 loss减少异常值影响
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
        'train_date': target_date # 记录模型名义日期
    }
    joblib.dump(model_data, model_path)
    
    val_score = model.best_score.get('valid', {}).get('rmse', 0.0)
    return f"{date_str}: Done (Val RMSE: {val_score:.4f})"


def main():
    print("=" * 60)
    print("LightGBM Training (Fixed Leakage)")
    print(f"Data range: {DATA_START_YEAR}-present")
    print(f"Training start: {TRAINING_START_YEAR}")
    print(f"Leakage Protection Gap: {GAP_DAYS} days (for 20-day target)")
    print(f"Time Decay Half-Life: {WEIGHT_HALF_LIFE} days")
    print("=" * 60)
    
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    print(f"Total files: {len(files)}")
    
    print("\nStep 1: Loading data...")
    all_data = load_all_data(files)
    if all_data is None:
        print("No data!")
        return
    print(f"Samples: {len(all_data)}")
    print(f"Date range: {all_data['date'].min()} to {all_data['date'].max()}")
    
    print("\nStep 2: Adding rank features...")
    all_data, rank_features = add_cross_sectional_rank(all_data)
    
    print("\nStep 3: Training models...")
    
    all_dates = sorted(all_data['date'].unique())
    
    # 从 TRAINING_START_YEAR 开始，每10天训练一个模型
    start_date = pd.Timestamp(f"{TRAINING_START_YEAR}-01-01")
    end_date = all_data['date'].max() - pd.Timedelta(days=5) # 留一点余量
    
    train_dates = []
    current_date = start_date
    while current_date <= end_date:
        # 找到最近的交易日
        valid_dates = [d for d in all_dates if d >= current_date]
        if valid_dates:
            train_dates.append(valid_dates[0])
        current_date += pd.Timedelta(days=10)
    
    train_dates = sorted(list(set(train_dates)))
    
    print(f"Total models to train: {len(train_dates)}")
    
    for target_date in tqdm(train_dates, desc="Training"):
        try:
            # 检查是否有文件存在（断点续传）
            date_str = target_date.strftime('%Y%m%d')
            model_path = os.path.join(MODEL_DIR, f"lgbm_{date_str}.pkl")
            if os.path.exists(model_path):
                continue
                
            train_model_for_date(target_date, all_data, rank_features)
        except Exception as e:
            print(f"Error {target_date}: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
