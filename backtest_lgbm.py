#!/usr/bin/env python3
"""
LightGBM 回测脚本
包含以下特性：
1. 月度滚动模型 (Monthly Rolling)
2. T+1 交易执行 (T日使用T-1信号)
3. 换手率控制 (Buffer逻辑: Keep Top 30, Target Top 20)
4. 真实交易限制 (停牌无法交易，估值使用最近价格)
5. 仓位控制 (单股最大权重 5%)
6. 费率控制 (万五佣金，最低5元)
7. 流动性与涨跌停限制 (Limit Up/Down, Max Amount Ratio)
8. IC 分析 (Rank IC, IR, Decay)
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ 配置 ============
DAILY_DIR = "data/daily"
MODEL_DIR = "models_lgbm"

# 时间范围 (对应 Monthly Training)
START_YEAR = 2003
END_YEAR = 2025

# 选股参数
TOP_N = 20           # 目标持仓数量
KEEP_TOP_N = 30      # 缓冲期：排名跌出30才卖出（降低换手）
REBALANCE_DAYS = 5   # 调仓频率 (交易日)

# 资金与风控
INITIAL_CASH = 1_000_000
COMMISSION = 0.0005  # 单边万五
MIN_COMMISSION = 5   # 最低五元
MAX_WEIGHT = 0.05    # 单只股票最大权重 5%
MAX_AMOUNT_RATIO = 0.02 # 最大成交额占比 (2% of daily volume)

# 基础特征 (必须与训练时一致)
BASE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_40', 'ret_60',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio',
    'volatility_5', 'volatility_10', 'volatility_20',
    'inv_volatility_5', 'inv_volatility_10', 'inv_volatility_20',
    'vol_ratio', 'vol_ratio_20',
    'new_high_ratio',
]

# 股票筛选条件
MIN_PRICE = 2.0
MAX_PRICE = 1000.0
MIN_AVG_VOLUME = 1000000
MIN_LISTING_DAYS = 60


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
    """计算原始特征"""
    df = df.copy()
    
    # 价格过滤
    df = df[df['close'] > 0]
    df = df[df['volume'] > 0]
    
    # 保留原始涨跌幅用于回测限制 (未裁剪)
    df['raw_pct_change'] = df['close'].pct_change(1)
    
    # 收益率 - 带裁剪 (用于模型特征)
    df['ret_1'] = df['raw_pct_change'].clip(-0.2, 0.2)
    df['ret_5'] = df['close'].pct_change(5).clip(-0.5, 0.5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20).clip(-1, 1)
    df['ret_40'] = df['close'].pct_change(40)
    df['ret_60'] = df['close'].pct_change(60)
    
    # 均线比率 - 带裁剪
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    df['ma5_ratio'] = (df['close'] / df['ma5']).clip(0.5, 2)
    df['ma10_ratio'] = (df['close'] / df['ma10']).clip(0.5, 2)
    df['ma20_ratio'] = (df['close'] / df['ma20']).clip(0.5, 2)
    
    # 波动率 - 增加稳健性 (避免除零)
    df['volatility_5'] = df['ret_1'].rolling(5).std().clip(lower=1e-4)
    df['volatility_10'] = df['ret_1'].rolling(10).std().clip(lower=1e-4)
    df['volatility_20'] = df['ret_1'].rolling(20).std().clip(lower=1e-4)
    
    df['inv_volatility_5'] = 1 / df['volatility_5']
    df['inv_volatility_10'] = 1 / df['volatility_10']
    df['inv_volatility_20'] = 1 / df['volatility_20']
    
    # 成交量
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    df['vol_ratio_20'] = df['volume'] / df['vol_ma20']
    
    # 250日新高 - 带裁剪
    df['high_250'] = df['close'].rolling(250).max()
    df['new_high_ratio'] = (df['close'] / df['high_250']).clip(0, 2)
    
    # 计算未来收益（用于计算 IC）
    df['future_ret_5'] = df['close'].shift(-5) / df['close'] - 1
    
    return df


def load_and_merge_all_data():
    """
    加载所有股票数据
    """
    files = glob.glob(os.path.join(DAILY_DIR, "*.parquet"))
    
    all_data = []
    valid_symbols = []
    
    print("Loading and processing all stocks...")
    for f in tqdm(files):
        symbol = os.path.splitext(os.path.basename(f))[0]
        
        try:
            df = pd.read_parquet(f)
            if len(df) < 100:
                continue
            
            if not filter_stock(df):
                continue
            
            df = compute_features(df)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            
            # 保留列：加入 amount 和 raw_pct_change 用于回测限制
            cols = ['date', 'symbol'] + BASE_FEATURES + ['close', 'amount', 'raw_pct_change', 'future_ret_5']
            # 注意：akshare 数据可能有 'amount' 或 'turnover' (成交额)
            # 假设 df 中已有 'amount' (成交额)
            if 'amount' not in df.columns and 'turnover' in df.columns:
                 # 有些源可能是 turnover 代表换手率，成交额叫 amount
                 # 这里假设 input parquet 有 amount
                 pass

            df = df[cols].dropna()
            
            if len(df) > 0:
                all_data.append(df)
                valid_symbols.append(symbol)
                
        except Exception as e:
            pass
    
    # 合并所有数据
    print("Merging data...")
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.sort_values(['date', 'symbol'])
    
    print(f"Total records: {len(merged)}")
    print(f"Valid symbols: {len(valid_symbols)}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged, valid_symbols


def generate_signals(all_data):
    """
    生成预测信号 (Monthly Rolling)
    读取 models_lgbm/lgbm_YYYYMM.pkl
    """
    all_predictions = []
    
    # 获取数据中的所有月份
    all_data['year_month'] = all_data['date'].dt.to_period('M')
    months = sorted(all_data['year_month'].unique())
    
    for ym in tqdm(months, desc="Generating signals"):
        year = ym.year
        month = ym.month
        
        if year < START_YEAR: 
            continue
            
        # 尝试加载对应的月度模型
        model_path = os.path.join(MODEL_DIR, f"lgbm_{year}{month:02d}.pkl")
        if not os.path.exists(model_path):
            continue
        
        # 加载模型
        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_cols = model_data['features']
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue
        
        # 获取该月的数据
        month_data = all_data[(all_data['date'].dt.year == year) & (all_data['date'].dt.month == month)].copy()
        
        if len(month_data) == 0:
            continue
        
        # 每天做预测
        dates = sorted(month_data['date'].unique())
        
        for date in dates:
            day_data = month_data[month_data['date'] == date].copy()
            
            if len(day_data) < 10:
                continue
            
            # 横截面 rank (关键)
            for col in BASE_FEATURES:
                day_data[f'{col}_rank'] = day_data[col].rank(pct=True, method='average')
            
            # 预测
            X = day_data[[c for c in feature_cols if c in day_data.columns]]
            if len(X) > 0:
                day_data['pred_ret'] = model.predict(X)
                all_predictions.append(day_data[['date', 'symbol', 'pred_ret', 'close', 'future_ret_5']])
    
    if not all_predictions:
        print("No predictions generated!")
        return pd.DataFrame()

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(['date', 'pred_ret'], ascending=[True, False])
    
    return predictions


def build_data_matrices(all_data, valid_symbols):
    """构建数据矩阵（高效查询）"""
    print("Building data matrices...")
    
    price_df = all_data.pivot(index='date', columns='symbol', values='close').sort_index()
    amount_df = all_data.pivot(index='date', columns='symbol', values='amount').sort_index()
    pct_change_df = all_data.pivot(index='date', columns='symbol', values='raw_pct_change').sort_index()
    
    # 计算昨收 (Pre Close) 用于 Limit 计算 (可选，这里直接用 pct_change 判断涨跌停)
    # pre_close_df = price_df.shift(1)
    # 为了接口兼容，返回 pre_close_df (虽然这里暂用 pct_change 判断)
    pre_close_df = price_df.shift(1)
    
    return price_df, pre_close_df, amount_df, pct_change_df


def analyze_ic(predictions):
    """
    计算 IC 和 ICIR
    """
    if predictions.empty:
        return None

    print("Calculating IC metrics...")
    # 每天计算 Rank IC (Spearman Correlation)
    ic_series = predictions.groupby('date').apply(
        lambda x: x['pred_ret'].corr(x['future_ret_5'], method='spearman')
    )
    
    # 移除 NaN
    ic_series = ic_series.dropna()
    
    if len(ic_series) == 0:
        return None

    # 滚动 IC
    rolling_ic_20 = ic_series.rolling(20).mean()
    rolling_ic_60 = ic_series.rolling(60).mean()
    
    # 累积 IC
    cumulative_ic = ic_series.cumsum()
    
    # 指标
    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    icir = mean_ic / ic_std if ic_std != 0 else 0
    positive_ratio = (ic_series > 0).mean()
    
    print("\n" + "=" * 60)
    print("IC ANALYSIS")
    print("=" * 60)
    print(f"Mean IC:          {mean_ic:.4f}")
    print(f"IC Std:           {ic_std:.4f}")
    print(f"ICIR:             {icir:.4f}")
    print(f"Positive IC Ratio:{positive_ratio:.2%}")
    print("=" * 60)
    
    return ic_series, rolling_ic_20, rolling_ic_60, cumulative_ic


def backtest(signals, price_df, pre_close_df, amount_df, pct_change_df):
    """
    执行回测
    Args:
        signals: 预测信号
        price_df: 收盘价矩阵
        pre_close_df: 昨收矩阵
        amount_df: 成交额矩阵
        pct_change_df: 涨跌幅矩阵
    """
    if signals.empty:
        print("No signals to backtest!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 准备信号索引
    signals_by_date = signals.set_index('date')
    
    # 交易日历
    all_dates = sorted(price_df.index)
    start_date = signals['date'].min()
    end_date = signals['date'].max()
    
    # 确定调仓日
    trading_days = [d for d in all_dates if d >= start_date and d <= end_date]
    rebalance_dates = trading_days[::REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)
    
    print(f"Rebalance dates: {len(rebalance_dates)} (every {REBALANCE_DAYS} trading days)")
    
    # 初始化
    cash = INITIAL_CASH
    holdings = {}  # symbol -> shares
    portfolio_values = []
    trades = []
    
    # 全局价格缓存（用于停牌估值）
    last_known_prices = {}
    
    # 用于计算换手率
    turnover_data = [] 
    
    for i, date in enumerate(tqdm(trading_days, desc="Backtesting")):
        date_ts = pd.Timestamp(date)
        
        # --- 1. 获取当日数据 ---
        tradable_prices = {} # 可交易价格
        day_prices = {}      # 所有价格 (包含停牌但有旧价格)
        day_amounts = {}     # 成交额
        day_pcts = {}        # 涨跌幅
        
        if date_ts in price_df.index:
            today_row = price_df.loc[date_ts]
            valid_prices = today_row[today_row.notna() & (today_row > 0)]
            
            # 更新缓存
            for sym, price in valid_prices.items():
                last_known_prices[sym] = price
            
            tradable_prices = valid_prices.to_dict()
            
            # 附加数据
            if date_ts in amount_df.index:
                day_amounts = amount_df.loc[date_ts].to_dict()
            if date_ts in pct_change_df.index:
                day_pcts = pct_change_df.loc[date_ts].to_dict()
        
        # --- 2. 每日估值 ---
        portfolio_value = cash
        holdings_value = 0
        
        for sym, shares in holdings.items():
            # 优先用今日交易价，否则用最近价格
            price = tradable_prices.get(sym, last_known_prices.get(sym, 0))
            if price > 0:
                val = shares * price
                portfolio_value += val
                holdings_value += val
        
        # --- 3. 调仓逻辑 ---
        # T+1 逻辑
        if date_ts in rebalance_set and i > 0:
            prev_date = trading_days[i-1]
            
            # 获取 T-1 的所有信号
            if prev_date in signals_by_date.index:
                day_signals = signals_by_date.loc[prev_date]
                if isinstance(day_signals, pd.Series):
                    day_signals = day_signals.to_frame().T
                
                # 按预测收益排序
                day_signals = day_signals.sort_values('pred_ret', ascending=False)
                
                # --- Buffer 逻辑 ---
                current_symbols = list(holdings.keys())
                kept_symbols = []
                
                ranked_symbols = day_signals['symbol'].tolist()
                symbol_rank = {sym: r for r, sym in enumerate(ranked_symbols)}
                
                # 检查现有持仓
                for sym in current_symbols:
                    rank = symbol_rank.get(sym, 9999)
                    if rank < KEEP_TOP_N:
                        kept_symbols.append(sym)
                
                # 补齐到 TOP_N
                final_symbols = list(kept_symbols)
                for sym in ranked_symbols:
                    if len(final_symbols) >= TOP_N:
                        break
                    if sym not in final_symbols:
                        final_symbols.append(sym)
                
                # --- 调仓前市值 ---
                pre_trade_nav = portfolio_value
                
                # --- 目标权重 ---
                n_stocks = len(final_symbols)
                if n_stocks > 0:
                    equal_weight = 1.0 / n_stocks
                    target_weight = min(equal_weight, MAX_WEIGHT)
                    if target_weight * n_stocks < 0.95 and n_stocks < TOP_N:
                         target_weight = 1.0 / n_stocks
                else:
                    target_weight = 0
                
                # --- 交易执行 ---
                buy_val_total = 0
                sell_val_total = 0
                
                # 1. 卖出 (不在目标池)
                for sym in list(holdings.keys()):
                    if sym not in final_symbols:
                        # 检查可交易性
                        if sym not in tradable_prices: continue # 停牌
                        
                        # 检查跌停 (Limit Down): 一般跌停无法卖出
                        # A股跌停约 -10% (-0.098)
                        pct = day_pcts.get(sym, 0)
                        if pct < -0.098:
                            continue # 跌停锁死，无法卖出
                        
                        price = tradable_prices[sym]
                        shares = holdings[sym]
                        
                        # 流动性限制: 卖出量不超过当日成交额的一定比例?
                        # 实际上卖出通常比较刚性，这里仅对买入做严格限制，卖出主要看跌停
                        # 但如果要加上 MAX_AMOUNT_RATIO:
                        day_amt = day_amounts.get(sym, 0)
                        max_sell_val = day_amt * MAX_AMOUNT_RATIO if day_amt > 0 else 0
                        
                        sell_val = shares * price
                        
                        # 如果要卖出的量 > 流动性限制，只能卖一部分
                        if sell_val > max_sell_val and max_sell_val > 0:
                            shares_to_sell = int(max_sell_val / price / 100) * 100
                        else:
                            shares_to_sell = shares
                        
                        if shares_to_sell <= 0: continue

                        amount = shares_to_sell * price
                        fee = max(amount * COMMISSION, MIN_COMMISSION)
                        proceeds = amount - fee
                        
                        cash += proceeds
                        holdings[sym] -= shares_to_sell
                        if holdings[sym] == 0:
                            del holdings[sym]
                        
                        sell_val_total += amount
                        trades.append({
                            'date': date, 'symbol': sym, 'action': 'SELL',
                            'shares': shares_to_sell, 'price': price, 'fee': fee
                        })
                
                # 2. 买入/调仓
                # 计算当前总资产 (现金+持仓)
                current_total_value = cash
                for sym in holdings:
                    price = tradable_prices.get(sym, last_known_prices.get(sym, 0))
                    current_total_value += holdings[sym] * price
                
                target_value_per_stock = current_total_value * target_weight
                
                for sym in final_symbols:
                    if sym not in tradable_prices: continue
                    
                    price = tradable_prices[sym]
                    if price <= 0: continue
                    
                    # 检查涨停 (Limit Up): 无法买入
                    pct = day_pcts.get(sym, 0)
                    if pct > 0.098:
                        continue 
                        
                    current_shares = holdings.get(sym, 0)
                    current_stock_value = current_shares * price
                    
                    diff_value = target_value_per_stock - current_stock_value
                    
                    # 阈值
                    if abs(diff_value) < max(target_value_per_stock * 0.1, 2000):
                        continue
                        
                    # 流动性限制
                    day_amt = day_amounts.get(sym, 0)
                    max_trade_val = day_amt * MAX_AMOUNT_RATIO if day_amt > 0 else 0
                    
                    if diff_value > 0:
                        # 买入
                        available_cash = max(0, cash)
                        max_buy_value = available_cash / (1 + COMMISSION)
                        buy_value = min(diff_value, max_buy_value)
                        
                        # 流动性限制买入
                        if buy_value > max_trade_val and max_trade_val > 0:
                            buy_value = max_trade_val
                        
                        shares_to_buy = int(buy_value / price / 100) * 100
                        if shares_to_buy > 0:
                            cost = shares_to_buy * price
                            fee = max(cost * COMMISSION, MIN_COMMISSION)
                            if cash >= cost + fee:
                                cash -= (cost + fee)
                                holdings[sym] = holdings.get(sym, 0) + shares_to_buy
                                buy_val_total += cost
                                trades.append({
                                    'date': date, 'symbol': sym, 'action': 'BUY',
                                    'shares': shares_to_buy, 'price': price, 'fee': fee
                                })
                                
                    elif diff_value < 0:
                        # 卖出平衡
                        # 检查跌停
                        if pct < -0.098: continue
                        
                        sell_value = abs(diff_value)
                        
                        # 流动性限制卖出
                        if sell_value > max_trade_val and max_trade_val > 0:
                            sell_value = max_trade_val
                            
                        shares_to_sell = int(sell_value / price / 100) * 100
                        if shares_to_sell > 0:
                            shares_to_sell = min(shares_to_sell, current_shares)
                            revenue = shares_to_sell * price
                            fee = max(revenue * COMMISSION, MIN_COMMISSION)
                            cash += (revenue - fee)
                            
                            holdings[sym] -= shares_to_sell
                            if holdings[sym] == 0:
                                del holdings[sym]
                                
                            sell_val_total += revenue
                            trades.append({
                                'date': date, 'symbol': sym, 'action': 'SELL_REBALANCE',
                                'shares': shares_to_sell, 'price': price, 'fee': fee
                            })
                            
                # --- 记录换手率 ---
                if pre_trade_nav > 0:
                    total_traded = buy_val_total + sell_val_total
                    turnover_rate = total_traded / (2 * pre_trade_nav)
                    turnover_data.append({
                        'date': date,
                        'turnover': turnover_rate,
                        'buy': buy_val_total,
                        'sell': sell_val_total,
                        'nav': pre_trade_nav
                    })

        # 记录
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash,
            'holdings_value': holdings_value,
            'holdings_count': len(holdings)
        })
    
    # 构建月度统计 (Optional placeholder)
    monthly_summary = pd.DataFrame() 
    
    return pd.DataFrame(portfolio_values), pd.DataFrame(trades), pd.DataFrame(turnover_data), monthly_summary


def calculate_metrics(df, turnover_df=None):
    """计算回测指标"""
    if df.empty:
        return {}
        
    df = df.sort_values('date').copy()
    df['returns'] = df['value'].pct_change()
    
    total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
    
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    volatility = df['returns'].std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
    
    df['cummax'] = df['value'].cummax()
    df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()
    
    win_rate = (df['returns'] > 0).mean()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 换手率
    turnover_metrics = {}
    if turnover_df is not None and len(turnover_df) > 0:
        turnover_df = turnover_df.copy()
        turnover_df['year'] = pd.to_datetime(turnover_df['date']).dt.year
        yearly_turnover = turnover_df.groupby('year')['turnover'].sum()
        turnover_metrics['avg_annual_turnover'] = yearly_turnover.mean()
        turnover_metrics['yearly_turnover'] = yearly_turnover.to_dict()
    else:
        turnover_metrics['avg_annual_turnover'] = 0
        turnover_metrics['yearly_turnover'] = {}
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar,
        'final_value': df['value'].iloc[-1],
        'years': years,
        **turnover_metrics
    }


def plot_results(df, ic_data=None):
    """绘图"""
    if df.empty: return

    rows = 4 if ic_data else 3
    fig, axes = plt.subplots(rows, 1, figsize=(14, 4 * rows))
    
    # 1. Value
    ax1 = axes[0]
    ax1.plot(df['date'], df['value'], label='Portfolio', color='blue')
    ax1.set_title('Portfolio Value')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=INITIAL_CASH, color='gray', linestyle='--')
    
    # 2. Drawdown
    ax2 = axes[1]
    df['cummax'] = df['value'].cummax()
    df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
    ax2.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # 3. Holdings
    ax3 = axes[2]
    ax3.plot(df['date'], df['holdings_count'], label='Holdings', color='green')
    ax3.set_title('Holdings Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. IC
    if ic_data:
        ic_series, r20, r60, cum_ic = ic_data
        ax4 = axes[3]
        ax4.bar(ic_series.index, ic_series.values, color='gray', alpha=0.3, label='Daily IC')
        ax4.plot(r20.index, r20.values, color='blue', label='MA20')
        ax4.plot(r60.index, r60.values, color='orange', label='MA60')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(cum_ic.index, cum_ic.values, color='green', linestyle='--', label='Cum IC')
        ax4.set_title('IC Analysis')
        ax4.legend(loc='upper left')
        
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    print("Saved plot to backtest_results.png")


def main():
    print("=" * 60)
    print("LightGBM Backtest - Limit/Amount Constraints")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  TOP_N: {TOP_N}, Buffer: {KEEP_TOP_N}")
    print(f"  REBALANCE_DAYS: {REBALANCE_DAYS}")
    print(f"  COMMISSION: {COMMISSION:.2%} (min {MIN_COMMISSION})")
    print(f"  MAX_WEIGHT: {MAX_WEIGHT:.0%}")
    print(f"  MAX_AMOUNT_RATIO: {MAX_AMOUNT_RATIO:.0%}")
    print(f"  INITIAL_CASH: {INITIAL_CASH:,.0f}")
    print("=" * 60)
    
    print("\nStep 1: Loading all data...")
    all_data, valid_symbols = load_and_merge_all_data()
    
    print("\nStep 2: Generating signals...")
    signals = generate_signals(all_data)
    print(f"Generated {len(signals)} signals")
    
    print("\nStep 3: Building data matrices...")
    price_df, pre_close_df, amount_df, pct_change_df = build_data_matrices(all_data, valid_symbols)
    
    print("\nStep 3.5: Analyzing IC...")
    ic_data = analyze_ic(signals)
    
    del all_data
    import gc
    gc.collect()
    
    print("\nStep 4: Running backtest...")
    results, trades, turnover_data, monthly_summary = backtest(
        signals, price_df, pre_close_df, amount_df, pct_change_df
    )
    
    print("\nStep 5: Calculating metrics...")
    metrics = calculate_metrics(results, turnover_data)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:           {metrics['years']:.1f} years")
    print(f"Initial Value:    {INITIAL_CASH:>15,.0f}")
    print(f"Final Value:      {metrics['final_value']:>15,.0f}")
    print(f"Total Return:     {metrics['total_return']:>15.2%}")
    print(f"Annual Return:    {metrics['annual_return']:>15.2%}")
    print(f"Volatility:       {metrics['volatility']:>15.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>15.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>15.2%}")
    print(f"Win Rate:         {metrics['win_rate']:>15.2%}")
    print(f"Calmar Ratio:     {metrics['calmar_ratio']:>15.2f}")
    print("-" * 60)
    print(f"Avg Annual Turnover: {metrics.get('avg_annual_turnover', 0):>10.2%}")
    print("=" * 60)
    
    # Save
    results.to_csv('backtest_results.csv', index=False)
    trades.to_csv('backtest_trades.csv', index=False)
    if not turnover_data.empty:
        turnover_data.to_csv('backtest_turnover.csv', index=False)
    
    plot_results(results, ic_data)


if __name__ == "__main__":
    main()
