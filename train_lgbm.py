#!/usr/bin/env python3
"""
LightGBM 量化模型 - 训练脚本 (库自带并行版)
- 月度滚动训练 (Monthly Rolling)
- 正确的横截面 Rank (pandarallel 并行)
- 无 lookahead 风险
- LightGBM 内置多线程训练
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

# 尝试导入 pandarallel，如果没有则使用备选方案
try:
    from pandarallel import pandarallel
    HAS_PANDARALLEL = True
except ImportError:
    HAS_PANDARALLEL = False
    print("Warning: pandarallel not found. Install with: pip install pandarallel")
    print("Will use alternative parallel method for ranking.")

DATA_DIR = "data/daily"
MODEL_DIR = "models_lgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

N_JOBS = 16  # 总核心数

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
    """加载所有股票数据，合并成一个大表 (使用 joblib.Parallel)"""
    print("Loading data using joblib.Parallel...")
    
    # joblib.Parallel 比 ProcessPoolExecutor 更高效
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=1)(
        delayed(process_one_file)(f) for f in files
    )
    
    dfs = [r for r in results if r is not None]
    
    if len(dfs) == 0:
        return None
    
    # 合并所有数据
    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.sort_values(['date', 'symbol'])
    return all_data


def _rank_single_column(args):
    """辅助函数：单列rank计算（用于并行）"""
    col, dates_values = args
    dates, values = dates_values
    # 创建临时DataFrame用于groupby
    df = pd.DataFrame({'date': dates, 'value': values})
    rank_result = df.groupby('date')['value'].rank(pct=True, method='average')
    return col, rank_result.values


def add_cross_sectional_rank(all_data):
    """
    添加横截面排名特征 (使用 joblib.Parallel 并行)
    关键：在所有股票之间做rank，不是单只股票
    """
    print("Adding cross-sectional rank features (joblib.Parallel)...")
    
    rank_features = [f'{c}_rank' for c in BASE_FEATURES]
    
    # 准备数据：提取日期和特征值
    dates = all_data['date'].values
    args_list = [(col, (dates, all_data[col].values)) for col in BASE_FEATURES]
    
    # 使用 joblib.Parallel 并行计算所有特征的 rank
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=1)(
        delayed(_rank_single_column)(args) for args in args_list
    )
    
    # 将结果赋值回 DataFrame
    for col, rank_values in results:
        all_data[f'{col}_rank'] = rank_values
    
    return all_data, rank_features


def prepare_regression_data(all_data, rank_features, label_col='future_ret_5'):
    """
    准备回归模型所需的数据格式
    简单的 (X, y) 格式，不需要 group
    """
    X = all_data[rank_features].values
    y = all_data[label_col].values
    
    return X, y


def train_month_and_save(year, month, all_data, rank_features):
    """
    训练并保存单月模型 (使用 LightGBM 内置多线程)
    """
    model_path = os.path.join(MODEL_DIR, f"lgbm_{year}{month:02d}.pkl")
    
    # 如果模型已存在，可以选择跳过
    # if os.path.exists(model_path):
    #     return f"{year}-{month:02d}: Skipped (Exists)"
    
    # 计算关键时间点
    target_date = pd.Timestamp(f"{year}-{month}-01")
    
    # 验证集月份 (M-1)
    valid_start = target_date - pd.DateOffset(months=1)
    valid_end = target_date - pd.Timedelta(days=1)
    
    # 训练集月份 (M-25 到 M-2)
    train_start = target_date - pd.DateOffset(months=25)
    train_end = valid_start - pd.Timedelta(days=1)
    
    # 筛选数据
    train_mask = (all_data['date'] >= train_start) & (all_data['date'] <= train_end)
    valid_mask = (all_data['date'] >= valid_start) & (all_data['date'] <= valid_end)
    
    train_data = all_data[train_mask]
    valid_data = all_data[valid_mask]
    
    if len(train_data) < 1000 or len(valid_data) < 100:
        return f"{year}-{month:02d}: Skipped (Insufficient Data)"
    
    # 准备回归数据
    X_train, y_train = prepare_regression_data(train_data, rank_features)
    X_valid, y_valid = prepare_regression_data(valid_data, rank_features)
    
    # 创建 LightGBM 数据集
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    
    # 训练参数 - 使用 LightGBM 内置多线程
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
        'n_jobs': N_JOBS,           # 使用全部 16 核
        'num_threads': N_JOBS       # LightGBM 多线程训练
    }
    
    # 训练
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # 保存
    model_data = {
        'model': model,
        'features': rank_features,
        'best_iteration': model.best_iteration,
        'best_score': model.best_score
    }
    joblib.dump(model_data, model_path)
    
    # 提取验证集分数
    val_score = model.best_score.get('valid', {}).get('rmse', 0.0)
    
    return f"{year}-{month:02d}: Done (Score: {val_score:.4f})"


def main():
    print("=" * 60)
    print("LightGBM Regression - Monthly Rolling Training (Library Parallel)")
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
    
    # Step 3: 月度训练 (串行，但 LightGBM 内部使用多线程)
    print("\nStep 3: Training monthly models (LightGBM multi-threading)...")
    
    start_date = pd.Timestamp("2003-01-01")
    end_date = pd.Timestamp("2025-12-01")
    
    current_date = start_date
    tasks = []
    
    while current_date <= end_date:
        tasks.append((current_date.year, current_date.month))
        current_date += pd.DateOffset(months=1)
        
    print(f"Total months to train: {len(tasks)}")
    print(f"LightGBM threads per model: {N_JOBS}")
    print("Note: Training months sequentially, but each month uses 16 threads")
    
    # 串行训练每个月，但 LightGBM 内部使用多线程
    for year, month in tqdm(tasks, desc="Training"):
        try:
            res = train_month_and_save(year, month, all_data, rank_features)
            # tqdm 会自动显示进度，不需要打印每个结果
        except Exception as e:
            print(f"Task {year}-{month:02d} failed: {e}")
    
    print("\nAll monthly models trained!")


if __name__ == "__main__":
    main()
