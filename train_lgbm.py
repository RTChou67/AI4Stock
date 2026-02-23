#!/usr/bin/env python3
"""
LightGBM 量化模型 - 训练脚本
- 月度滚动训练 (Monthly Rolling)
- 正确的横截面 Rank
- 无 lookahead 风险
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import joblib
import gc

DATA_DIR = "data/daily"
MODEL_DIR = "models_lgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

N_JOBS = 16

# 股票筛选条件
MIN_PRICE = 2.0
MAX_PRICE = 1000.0
MIN_AVG_VOLUME = 1000000
MIN_LISTING_DAYS = 60

# 基础特征列表
BASE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_40', 'ret_60',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio',
    'volatility_5', 'volatility_10', 'volatility_20',
    'inv_volatility_5', 'inv_volatility_10', 'inv_volatility_20',
    'vol_ratio', 'vol_ratio_20',
    'new_high_ratio',
]


def filter_stock(df):
    """筛选股票"""
    if len(df) < MIN_LISTING_DAYS:
        return False
    
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return False
    
    if df['close'].isna().all() or df['volume'].isna().all():
        return False
    
    recent = df.tail(20).copy()
    recent = recent[recent['close'] > 0]
    if len(recent) < 10:
        return False
    
    avg_price = recent['close'].mean()
    if avg_price < MIN_PRICE or avg_price > MAX_PRICE:
        return False
    
    avg_volume = recent['volume'].mean()
    if pd.isna(avg_volume) or avg_volume < MIN_AVG_VOLUME:
        return False
    
    zero_volume_days = (recent['volume'] == 0).sum()
    if zero_volume_days > 5:
        return False
    
    return True


def compute_features(df):
    """计算原始特征（时序）- 单只股票"""
    df = df.copy()
    
    # 价格过滤：去除异常价格和成交量
    df = df[df['close'] > 0]
    df = df[df['volume'] > 0]
    
    # 收益率
    df['ret_1'] = df['close'].pct_change(1).clip(-0.2, 0.2)
    df['ret_5'] = df['close'].pct_change(5).clip(-0.5, 0.5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20).clip(-1, 1)
    df['ret_40'] = df['close'].pct_change(40)
    df['ret_60'] = df['close'].pct_change(60)
    
    # 均线比率
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    df['ma5_ratio'] = (df['close'] / df['ma5']).clip(0.5, 2)
    df['ma10_ratio'] = (df['close'] / df['ma10']).clip(0.5, 2)
    df['ma20_ratio'] = (df['close'] / df['ma20']).clip(0.5, 2)
    
    # 波动率 - 增加稳健性
    df['volatility_5'] = df['ret_1'].rolling(5).std().clip(lower=1e-4)
    df['volatility_10'] = df['ret_1'].rolling(10).std().clip(lower=1e-4)
    df['volatility_20'] = df['ret_1'].rolling(20).std().clip(lower=1e-4)
    
    # 波动率反转
    df['inv_volatility_5'] = 1 / df['volatility_5']
    df['inv_volatility_10'] = 1 / df['volatility_10']
    df['inv_volatility_20'] = 1 / df['volatility_20']
    
    # 成交量
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    df['vol_ratio_20'] = df['volume'] / df['vol_ma20']
    
    # 250日新高
    df['high_250'] = df['close'].rolling(250).max()
    df['new_high_ratio'] = (df['close'] / df['high_250']).clip(0, 2)
    
    # 未来收益（标签）- 重新引入 clipping 避免极端值影响
    df['future_ret_5'] = (df['close'].shift(-5) / df['close'] - 1).clip(-0.2, 0.2)
    
    return df


def process_one_file(filepath):
    """处理单个文件，返回该股票的所有历史数据"""
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 100:
            return None
        
        if not filter_stock(df):
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df = compute_features(df)
        
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        df['symbol'] = symbol
        
        # 选择需要的列（注意：这里还没有做横截面rank！）
        cols = ['date', 'symbol'] + BASE_FEATURES + ['future_ret_5']
        df = df[cols].dropna()
        
        if len(df) < 60:
            return None
            
        return df
        
    except Exception as e:
        return None


def load_all_data(files):
    """加载所有股票数据，合并成一个大表"""
    dfs = []
    
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(process_one_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            df = future.result()
            if df is not None:
                dfs.append(df)
    
    if len(dfs) == 0:
        return None
    
    # 合并所有数据
    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.sort_values(['date', 'symbol'])
    return all_data


def add_cross_sectional_rank(all_data):
    """
    添加横截面排名特征
    关键：在所有股票之间做rank，不是单只股票
    """
    print("Adding cross-sectional rank features...")
    
    rank_features = []
    for col in tqdm(BASE_FEATURES, desc="Ranking"):
        rank_col = f'{col}_rank'
        # 按日期分组，每天内部做rank（这才是真正的横截面rank！）
        all_data[rank_col] = all_data.groupby('date')[col].transform(
            lambda x: x.rank(pct=True, method='average')
        )
        rank_features.append(rank_col)
    
    return all_data, rank_features


def prepare_regression_data(all_data, rank_features, label_col='future_ret_5'):
    """
    准备回归模型所需的数据格式
    简单的 (X, y) 格式，不需要 group
    """
    X = all_data[rank_features].values
    y = all_data[label_col].values
    
    return X, y


def train_month_with_early_stopping(target_year, target_month, all_data, rank_features):
    """
    训练单月模型 (Rolling Window)
    - 预测目标：target_year-target_month
    - 训练数据：过去24个月
      - Train: [M-25, M-2] (23个月)
      - Valid: [M-1]    (1个月)
    """
    
    # 计算关键时间点
    target_date = pd.Timestamp(f"{target_year}-{target_month}-01")
    
    # 验证集月份 (M-1)
    valid_start = target_date - pd.DateOffset(months=1)
    valid_end = target_date - pd.Timedelta(days=1) # Valid直到上个月最后一天
    
    # 训练集月份 (M-25 到 M-2)
    train_start = target_date - pd.DateOffset(months=25)
    train_end = valid_start - pd.Timedelta(days=1)
    
    # 筛选数据
    # 注意：all_data['date'] 是 timestamp
    train_mask = (all_data['date'] >= train_start) & (all_data['date'] <= train_end)
    valid_mask = (all_data['date'] >= valid_start) & (all_data['date'] <= valid_end)
    
    train_data = all_data[train_mask]
    valid_data = all_data[valid_mask]
    
    # 检查数据量
    if len(train_data) < 1000 or len(valid_data) < 100:
        # print(f"  Insufficient data for {target_year}-{target_month:02d}")
        return None
    
    # print(f"  Train: {len(train_data)} ({train_start.date()} to {train_end.date()})")
    # print(f"  Valid: {len(valid_data)} ({valid_start.date()} to {valid_end.date()})")
    
    # 准备回归数据
    X_train, y_train = prepare_regression_data(train_data, rank_features)
    X_valid, y_valid = prepare_regression_data(valid_data, rank_features)
    
    # 创建 LightGBM 数据集
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    
    # 训练参数
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 64,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': 4, # 降低单模型并发，因为数据量小，且我们可能希望以后并行跑月份
    }
    
    # 训练
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000, # 月度更新频繁，轮数可以稍减，或者靠 early_stopping
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)] # 关闭详细日志
    )
    
    return model


def main():
    print("=" * 60)
    print("LightGBM Regression - Monthly Rolling Training")
    print("=" * 60)
    
    # 获取所有文件
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    print(f"Total files: {len(files)}")
    
    # Step 1: 加载所有数据
    print("\nStep 1: Loading all data...")
    all_data = load_all_data(files)
    
    if all_data is None:
        print("No data loaded!")
        return
    
    print(f"Total samples: {len(all_data)}")
    print(f"Date range: {all_data['date'].min()} to {all_data['date'].max()}")
    
    # Step 2: 添加横截面排名特征
    print("\nStep 2: Adding cross-sectional rank features...")
    all_data, rank_features = add_cross_sectional_rank(all_data)
    
    # Step 3: 月度训练
    print("\nStep 3: Training monthly models...")
    
    # 生成月份列表：2003-01 到 2025-12
    # 我们需要前24个月做第一次训练，所以从2003年开始
    start_date = pd.Timestamp("2003-01-01")
    end_date = pd.Timestamp("2025-12-01")
    
    current_date = start_date
    months_to_train = []
    while current_date <= end_date:
        months_to_train.append((current_date.year, current_date.month))
        current_date += pd.DateOffset(months=1)
        
    print(f"Total months to train: {len(months_to_train)}")
    
    for year, month in tqdm(months_to_train, desc="Training Months"):
        model_path = os.path.join(MODEL_DIR, f"lgbm_{year}{month:02d}.pkl")
        
        # 如果模型已存在，可以选择跳过
        # if os.path.exists(model_path):
        #     continue
            
        model = train_month_with_early_stopping(year, month, all_data, rank_features)
        
        if model is not None:
            model_data = {
                'model': model,
                'features': rank_features,
                'best_iteration': model.best_iteration,
                'best_score': model.best_score
            }
            joblib.dump(model_data, model_path)
    
    print("\nAll monthly models trained!")


if __name__ == "__main__":
    main()
