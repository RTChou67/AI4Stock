#!/usr/bin/env python3
"""
LightGBM 回测脚本 (Debugged & Robust)
配置：过去1年数据，预测20天收益，10天调仓
输入：直接从 data/features 读取预计算好的特征
特性：
1. 每年年初重置资金
2. 收益集中度分析 (Top-k Contribution)
3. 月度 IC/ICIR 时序分析
4. 特征分布漂移检测 (KS Test)
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy import stats 

warnings.filterwarnings('ignore')

# 修改输入目录为 features
DATA_DIR = "data/features"
MODEL_DIR = "models_lgbm"

TOP_N = 20
KEEP_TOP_N = 30
REBALANCE_DAYS = 10 

INITIAL_CASH = 1_000_000
COMMISSION = 0.00002
MIN_COMMISSION = 5
MAX_WEIGHT = 0.05
MAX_AMOUNT_RATIO = 0.02

# ==== 新特征列表 (与 gen_feature.py 一致) ====
BASE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_60',
    'dist_ma5', 'dist_ma20', 'dist_ma60', 'dist_ma120',
    'std_20', 'std_60', 'atr_14',
    'vol_ratio_5', 'vol_ratio_20',
    'rsi_14', 'macd_diff', 'hl_range',
    'dist_high_20', 'dist_low_20'
]

# 选取代表性特征进行漂移检测
DRIFT_CHECK_FEATURES = [
    'ret_20', 'std_20', 'dist_ma60', 'atr_14', 'rsi_14', 'macd_diff'
]


def load_and_merge_all_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    all_data = []
    
    print(f"Loading {len(files)} feature files from {DATA_DIR}...")
    for f in tqdm(files, desc="Loading"):
        try:
            df = pd.read_parquet(f)
            if len(df) < 100:
                continue
            
            # 补全 symbol
            if 'symbol' not in df.columns:
                symbol = os.path.splitext(os.path.basename(f))[0]
                df['symbol'] = symbol
                
            df['date'] = pd.to_datetime(df['date'])
            
            # 确保 label 列名一致
            if 'future_20d_ret' in df.columns:
                df['future_ret_20'] = df['future_20d_ret']
            
            # 筛选必要列
            # 注意：如果特征文件里没有 future_ret_20，这里会报错或被过滤
            # 我们允许缺失 label (预测时不需要 label)，但在回测统计 IC 时需要
            # 这里采取策略：尽量保留
            
            cols_to_keep = ['date', 'symbol', 'close', 'amount'] + BASE_FEATURES
            if 'future_ret_20' in df.columns:
                cols_to_keep.append('future_ret_20')
            
            # 检查特征列是否存在
            missing = [c for c in cols_to_keep if c not in df.columns]
            if missing:
                # print(f"Missing cols in {f}: {missing}")
                continue
                
            df = df[cols_to_keep].dropna(subset=BASE_FEATURES)
            if len(df) > 0:
                all_data.append(df)
        except:
            continue
    
    if not all_data:
        return pd.DataFrame(), []
    
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.sort_values(['date', 'symbol'])
    print(f"Records: {len(merged)}")
    return merged, merged['symbol'].unique().tolist()


# ==== 分析模块 ====

def analyze_feature_drift(all_data):
    print("\n" + "=" * 60)
    print("FEATURE DRIFT ANALYSIS (KS Test)")
    print("=" * 60)
    
    if all_data.empty: return

    all_data['year'] = all_data['date'].dt.year
    years = sorted(all_data['year'].unique())
    
    # 过滤年份：仅检查 2021-2025
    years = [y for y in years if 2021 <= y <= 2025]
    
    header = f"{'Year':<6} | {'Feature':<15} | {'KS Stat':>8} | {'P-Value':>10} | {'Drift?'}"
    print(header)
    print("-" * len(header))
    
    for year in years:
        year_data = all_data[all_data['year'] == year]
        if len(year_data) < 100: continue
        
        drift_count = 0
        for feat in DRIFT_CHECK_FEATURES:
            if feat not in all_data.columns: continue
            
            stat, pval = stats.ks_2samp(year_data[feat], all_data[feat])
            is_drift = "YES !!!" if pval < 0.01 else "No"
            if pval < 0.01: drift_count += 1
            
            print(f"{year:<6} | {feat:<15} | {stat:>8.4f} | {pval:>10.4f} | {is_drift}")
        print("-" * len(header))


def analyze_concentration(yearly_stock_pnl):
    print("\n" + "=" * 80)
    print("PnL CONCENTRATION ANALYSIS")
    print("=" * 80)
    
    print(f"{'Year':<6} | {'Total PnL':>12} | {'Top10 PnL':>12} | {'Top10 %':>8} | {'Top1 PnL':>12} | {'Top1 %':>8}")
    print("-" * 80)
    
    for year, pnl_map in sorted(yearly_stock_pnl.items()):
        if not pnl_map: continue
        
        series = pd.Series(pnl_map)
        total_pnl = series.sum()
        sorted_pnl = series.sort_values(ascending=False)
        
        top10_sum = sorted_pnl.head(10).sum()
        top1_sum = sorted_pnl.head(1).sum() if len(sorted_pnl) > 0 else 0
        
        if abs(total_pnl) < 1:
            ratio10, ratio1 = 0, 0
        else:
            ratio10 = top10_sum / total_pnl
            ratio1 = top1_sum / total_pnl
            
        print(f"{year:<6} | {total_pnl:>12.0f} | {top10_sum:>12.0f} | {ratio10:>8.2%} | {top1_sum:>12.0f} | {ratio1:>8.2%}")
    print("=" * 80)


def analyze_monthly_ic_plot(predictions):
    print("Analyzing Monthly IC...")
    if predictions.empty:
        print("Predictions are empty, skipping IC analysis.")
        return
        
    if 'future_ret_20' not in predictions.columns:
        print("No label 'future_ret_20' in predictions, skipping IC analysis.")
        return

    valid_preds = predictions.dropna(subset=['future_ret_20'])
    if len(valid_preds) == 0: 
        print("No valid predictions with labels, skipping IC analysis.")
        return

    daily_ic = valid_preds.groupby('date').apply(
        lambda x: x['pred_ret'].corr(x['future_ret_20'], method='spearman')
    )
    
    # 修复：使用 'ME' 替代 'M'
    monthly_ic_mean = daily_ic.resample('ME').mean()
    monthly_ic_std = daily_ic.resample('ME').std()
    monthly_icir = monthly_ic_mean / monthly_ic_std
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    color = ['red' if v < 0 else 'blue' for v in monthly_ic_mean.values]
    ax1.bar(monthly_ic_mean.index, monthly_ic_mean.values, color=color, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title("Monthly Mean IC (Spearman)")
    
    ax2.plot(monthly_icir.index, monthly_icir.values, marker='o', linestyle='-', color='green')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title("Monthly ICIR")
    
    plt.tight_layout()
    plt.savefig('backtest_ic_monthly.png', dpi=150)
    print("Saved backtest_ic_monthly.png")


# ==========================


def generate_signals(all_data):
    all_dates = sorted(all_data['date'].unique())
    model_files = glob.glob(os.path.join(MODEL_DIR, "lgbm_*.pkl"))
    model_dates = []
    for mf in model_files:
        try:
            basename = os.path.basename(mf)
            date_str = basename.replace('lgbm_', '').replace('.pkl', '')
            m_date = pd.to_datetime(date_str, format='%Y%m%d')
            model_dates.append({'date': m_date, 'path': mf})
        except:
            continue
    
    # 修复：如果列表为空，直接返回空DF，避免 KeyError: 'date'
    if not model_dates:
        print("No models found in models_lgbm/ !")
        return pd.DataFrame()
    
    model_info_df = pd.DataFrame(model_dates).sort_values('date').reset_index(drop=True)

    all_predictions = []
    current_model = None
    current_model_path = None
    current_features = None
    
    start_date = model_info_df['date'].min()
    predict_dates = [d for d in all_dates if d >= start_date]
    
    feature_mismatch_warned = False
    
    for date in tqdm(predict_dates, desc="Generating Signals"):
        valid_models = model_info_df[model_info_df['date'] < date]
        if valid_models.empty: continue
            
        latest_model_row = valid_models.iloc[-1]
        model_path = latest_model_row['path']
        
        if model_path != current_model_path:
            try:
                m_data = joblib.load(model_path)
                current_model = m_data['model']
                current_features = m_data['features']
                current_model_path = model_path
            except:
                continue
        
        if current_model is None: continue

        day_data = all_data[all_data['date'] == date].copy()
        if len(day_data) < 10: continue
            
        for col in BASE_FEATURES:
            day_data[f'{col}_rank'] = day_data[col].rank(pct=True, method='average')
            
        valid_cols = [c for c in current_features if c in day_data.columns]
        
        if len(valid_cols) != len(current_features): 
            if not feature_mismatch_warned:
                missing = list(set(current_features) - set(day_data.columns))
                print(f"\n[WARNING] Feature mismatch! Model needs {len(current_features)}, found {len(valid_cols)}.")
                print(f"Missing in data: {missing[:5]} ...")
                print("Skipping predictions until models are retrained or data is fixed.")
                feature_mismatch_warned = True
            continue
            
        X = day_data[current_features]
        if len(X) > 0:
            day_data['pred_ret'] = current_model.predict(X, num_threads=1)
            # 安全的列选择
            cols_to_save = ['date', 'symbol', 'pred_ret', 'close']
            if 'future_ret_20' in day_data.columns:
                cols_to_save.append('future_ret_20')
            else:
                day_data['future_ret_20'] = np.nan
                cols_to_save.append('future_ret_20')
                
            all_predictions.append(day_data[cols_to_save])

    if not all_predictions: 
        print("\n[ERROR] No predictions generated! Check feature mismatch warning above.")
        return pd.DataFrame()
        
    return pd.concat(all_predictions, ignore_index=True).sort_values(['date', 'pred_ret'], ascending=[True, False])


def build_data_matrices(all_data, valid_symbols):
    price_df = all_data.pivot(index='date', columns='symbol', values='close').sort_index()
    amount_df = all_data.pivot(index='date', columns='symbol', values='amount').sort_index()
    
    # 构建一字板标识: 如果最高价==最低价，说明今天全天没有波动（一字涨跌停或停牌）
    # 对于停牌股，价格会保持前收盘价，但通常 volume 为 0，所以在后续可能也被过滤
    # 这里我们严格标记 high == low 为不可交易 (True)
    if 'high' in all_data.columns and 'low' in all_data.columns:
        high_df = all_data.pivot(index='date', columns='symbol', values='high').sort_index()
        low_df = all_data.pivot(index='date', columns='symbol', values='low').sort_index()
        # 认为高低价非常接近的就是一字板
        limit_locked_df = abs(high_df - low_df) < 1e-4
    else:
        # Fallback 兼容
        limit_locked_df = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
        
    if 'ret_1' in all_data.columns:
        pct_change_df = all_data.pivot(index='date', columns='symbol', values='ret_1').sort_index()
    else:
        pct_change_df = price_df.pct_change()
        
    return price_df, price_df.shift(1), amount_df, pct_change_df, limit_locked_df


def calculate_yearly_ic_and_top10(predictions, year):
    if predictions.empty: return {'ic_mean': np.nan, 'ic_std': np.nan, 'top10_mean_ret': np.nan}
    
    year_data = predictions[predictions['date'].dt.year == year]
    if len(year_data) == 0:
        return {'ic_mean': np.nan, 'ic_std': np.nan, 'top10_mean_ret': np.nan}
    
    valid_data = year_data.dropna(subset=['future_ret_20'])
    if len(valid_data) == 0: return {'ic_mean': np.nan} 

    daily_ic = valid_data.groupby('date').apply(
        lambda x: x['pred_ret'].corr(x['future_ret_20'], method='spearman')
    ).dropna()
    
    daily_top_returns = []
    for date, day_data in valid_data.groupby('date'):
        if len(day_data) >= 10:
            top_stocks = day_data.nlargest(10, 'pred_ret')
            avg_ret = top_stocks['future_ret_20'].mean()
            daily_top_returns.append(avg_ret)
    
    ic_mean = daily_ic.mean() if len(daily_ic) > 0 else np.nan
    ic_std = daily_ic.std() if len(daily_ic) > 0 else np.nan
    top10_mean_ret = pd.Series(daily_top_returns).mean() if daily_top_returns else np.nan
    
    return {'ic_mean': ic_mean, 'ic_std': ic_std, 'top10_mean_ret': top10_mean_ret}


def backtest(signals, predictions, price_df, pre_close_df, amount_df, pct_change_df, limit_locked_df):
    if signals.empty:
        print("No signals!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    signals_by_date = signals.set_index('date')
    all_dates = sorted(price_df.index)
    start_date = signals['date'].min()
    end_date = signals['date'].max()
    
    trading_days = [d for d in all_dates if d >= start_date and d <= end_date]
    rebalance_indices = list(range(0, len(trading_days), REBALANCE_DAYS))
    rebalance_set = set([trading_days[i] for i in rebalance_indices])
    
    cash = INITIAL_CASH
    yearly_stock_pnl = {}
    current_stock_pnl = {}
    
    holdings = {} 
    holding_costs = {} 
    
    portfolio_values = []
    trades = []
    last_known_prices = {}
    turnover_data = []
    yearly_returns = []
    
    current_year_start_value = INITIAL_CASH
    current_year_max_value = INITIAL_CASH
    current_year_turnover = 0.0
    last_processed_date = None
    
    for i, date in enumerate(tqdm(trading_days, desc="Backtesting")):
        date_ts = pd.Timestamp(date)
        
        # 年度重置
        if last_processed_date is not None and date_ts.year != last_processed_date.year:
            prev_year = last_processed_date.year
            year_end_value = cash
            for sym, shares in holdings.items():
                price = last_known_prices.get(sym, 0)
                if price > 0:
                    val = shares * price
                    year_end_value += val
                    cost = holding_costs.get(sym, 0)
                    pnl = (price - cost) * shares
                    current_stock_pnl[sym] = current_stock_pnl.get(sym, 0) + pnl
            
            yearly_stock_pnl[prev_year] = current_stock_pnl.copy()
            year_return = (year_end_value / current_year_start_value) - 1
            year_max_drawdown = (current_year_max_value - year_end_value) / current_year_max_value if current_year_max_value > 0 else 0
            
            ic_stats = calculate_yearly_ic_and_top10(predictions, prev_year)
            yearly_returns.append({
                'year': prev_year, 'start_value': current_year_start_value, 'end_value': year_end_value,
                'return': year_return, 'max_drawdown': year_max_drawdown, 'turnover': current_year_turnover,
                **ic_stats
            })
            
            cash = INITIAL_CASH
            holdings = {}
            holding_costs = {}
            current_stock_pnl = {}
            current_year_start_value = INITIAL_CASH
            current_year_max_value = INITIAL_CASH
            current_year_turnover = 0.0
        
        # 价格更新
        tradable_prices = {}
        day_amounts = {}
        day_pcts = {}
        day_locked = {} # 记录一字板状态
        
        if date_ts in price_df.index:
            today_row = price_df.loc[date_ts]
            valid_prices = today_row[today_row.notna() & (today_row > 0)]
            for sym, price in valid_prices.items(): last_known_prices[sym] = price
            tradable_prices = valid_prices.to_dict()
            if date_ts in amount_df.index: day_amounts = amount_df.loc[date_ts].to_dict()
            if date_ts in pct_change_df.index: day_pcts = pct_change_df.loc[date_ts].to_dict()
            if date_ts in limit_locked_df.index: day_locked = limit_locked_df.loc[date_ts].to_dict()
        
        # 净值计算
        portfolio_value = cash
        for sym, shares in holdings.items():
            price = tradable_prices.get(sym, last_known_prices.get(sym, 0))
            if price > 0: portfolio_value += shares * price
        
        if portfolio_value > current_year_max_value: current_year_max_value = portfolio_value
        
        # 调仓
        if date_ts in rebalance_set:
            prev_trade_idx = i - 1
            if prev_trade_idx >= 0:
                prev_date = trading_days[prev_trade_idx]
                if prev_date in signals_by_date.index:
                    day_signals = signals_by_date.loc[prev_date]
                    if isinstance(day_signals, pd.Series): day_signals = day_signals.to_frame().T
                    day_signals = day_signals.sort_values('pred_ret', ascending=False)
                    
                    current_symbols = list(holdings.keys())
                    ranked_symbols = day_signals['symbol'].tolist()
                    symbol_rank = {sym: r for r, sym in enumerate(ranked_symbols)}
                    
                    kept_symbols = [sym for sym in current_symbols if symbol_rank.get(sym, 9999) < KEEP_TOP_N]
                    final_symbols = list(kept_symbols)
                    for sym in ranked_symbols:
                        if len(final_symbols) >= TOP_N: break
                        if sym not in final_symbols: final_symbols.append(sym)
                    
                    pre_trade_nav = portfolio_value
                    n_stocks = len(final_symbols)
                    target_weight = min(1.0 / n_stocks if n_stocks > 0 else 0, MAX_WEIGHT)
                    
                    buy_val_total = 0
                    sell_val_total = 0
                    
                    # Sell
                    for sym in list(holdings.keys()):
                        if sym not in final_symbols:
                            if sym not in tradable_prices: continue
                            if day_pcts.get(sym, 0) < -0.098: continue
                            if day_locked.get(sym, False): continue # 一字跌停无法卖出
                            
                            price = tradable_prices[sym]
                            shares = holdings[sym]
                            sell_val = shares * price
                            fee = max(sell_val * COMMISSION, MIN_COMMISSION)
                            cash += (sell_val - fee)
                            
                            cost = holding_costs.get(sym, 0)
                            pnl = (price - cost) * shares - fee
                            current_stock_pnl[sym] = current_stock_pnl.get(sym, 0) + pnl
                            
                            del holdings[sym]
                            del holding_costs[sym]
                            sell_val_total += sell_val
                            trades.append({'date': date, 'symbol': sym, 'action': 'SELL', 'shares': shares, 'price': price, 'fee': fee, 'pnl': pnl})
                    
                    # Buy
                    target_value_per_stock = portfolio_value * target_weight
                    for sym in final_symbols:
                        if sym not in tradable_prices: continue
                        price = tradable_prices[sym]
                        if price <= 0 or day_pcts.get(sym, 0) > 0.098: continue
                        if day_locked.get(sym, False): continue # 一字涨停无法买入
                        
                        current_shares = holdings.get(sym, 0)
                        diff_value = target_value_per_stock - (current_shares * price)
                        if abs(diff_value) < max(target_value_per_stock * 0.1, 2000): continue
                        
                        if diff_value > 0:
                            max_buy = cash / (1 + COMMISSION)
                            buy_val = min(diff_value, max_buy)
                            day_amt = day_amounts.get(sym, 0)
                            if day_amt > 0: buy_val = min(buy_val, day_amt * MAX_AMOUNT_RATIO)
                            
                            shares_buy = int(buy_val / price / 100) * 100
                            if shares_buy > 0:
                                cost = shares_buy * price
                                fee = max(cost * COMMISSION, MIN_COMMISSION)
                                if cash >= cost + fee:
                                    cash -= (cost + fee)
                                    prev_shares = holdings.get(sym, 0)
                                    prev_cost = holding_costs.get(sym, 0)
                                    new_shares = prev_shares + shares_buy
                                    new_avg_cost = (prev_shares * prev_cost + cost + fee) / new_shares
                                    holdings[sym] = new_shares
                                    holding_costs[sym] = new_avg_cost
                                    buy_val_total += cost
                                    trades.append({'date': date, 'symbol': sym, 'action': 'BUY', 'shares': shares_buy, 'price': price, 'fee': fee})
                    
                    if pre_trade_nav > 0:
                        total_traded = buy_val_total + sell_val_total
                        turnover_rate = total_traded / (2 * pre_trade_nav)
                        turnover_data.append({'date': date, 'turnover': turnover_rate})
                        current_year_turnover += turnover_rate
        
        portfolio_values.append({'date': date, 'value': portfolio_value, 'cash': cash, 'holdings_count': len(holdings)})
        last_processed_date = date_ts
    
    # 最后一年
    if len(trading_days) > 0 and last_processed_date is not None:
        last_date = pd.Timestamp(trading_days[-1])
        final_value = cash
        for sym, shares in holdings.items():
            price = last_known_prices.get(sym, 0)
            if price > 0:
                final_value += shares * price
                cost = holding_costs.get(sym, 0)
                pnl = (price - cost) * shares
                current_stock_pnl[sym] = current_stock_pnl.get(sym, 0) + pnl
        
        yearly_stock_pnl[last_date.year] = current_stock_pnl.copy()
        ic_stats = calculate_yearly_ic_and_top10(predictions, last_date.year)
        yearly_returns.append({
            'year': last_date.year, 'start_value': current_year_start_value, 'end_value': final_value,
            'return': (final_value / current_year_start_value) - 1, 'max_drawdown': 0, 
            'turnover': current_year_turnover, **ic_stats
        })
    
    return pd.DataFrame(portfolio_values), pd.DataFrame(trades), pd.DataFrame(turnover_data), pd.DataFrame(yearly_returns), yearly_stock_pnl


def calculate_metrics(yearly_df):
    if yearly_df.empty: return {}
    rets = yearly_df['return'].values
    return {
        'years': len(rets),
        'annual_return': rets.mean(),
        'sharpe_ratio': (rets.mean() - 0.02) / rets.std() if rets.std() > 0 else 0,
        'max_drawdown': yearly_df['max_drawdown'].max()
    }


def plot_results(df):
    if df.empty: return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(df['date'], df['value'], label='Portfolio', color='blue')
    axes[0].set_title('Portfolio Value')
    
    df['dd'] = (df['value'] - df['value'].cummax()) / df['value'].cummax()
    axes[1].fill_between(df['date'], df['dd'], 0, color='red', alpha=0.3)
    axes[1].set_title('Drawdown')
    
    axes[2].plot(df['date'], df['holdings_count'], color='green')
    axes[2].set_title('Holdings Count')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    print("Saved backtest_results.png")


def main():
    print("=" * 60)
    print("LightGBM Backtest - Advanced Analysis")
    print("=" * 60)
    
    print("\nStep 1: Loading feature data...")
    all_data, valid_symbols = load_and_merge_all_data()
    
    # 1. 特征漂移分析
    analyze_feature_drift(all_data)
    
    print("\nStep 2: Generating signals...")
    signals = generate_signals(all_data)
    
    if signals.empty:
        print("[STOP] No signals generated. Please check feature mismatch warning above.")
        return
    
    print("\nStep 3: Building matrices...")
    price_df, pre_close_df, amount_df, pct_change_df, limit_locked_df = build_data_matrices(all_data, valid_symbols)
    
    # 2. 月度 IC 分析
    analyze_monthly_ic_plot(signals)
    
    del all_data
    import gc
    gc.collect()
    
    print("\nStep 4: Running backtest...")
    results, trades, turnover_data, yearly_summary, yearly_stock_pnl = backtest(
        signals, signals, price_df, pre_close_df, amount_df, pct_change_df, limit_locked_df
    )
    
    print("\nStep 5: Analysis...")
    metrics = calculate_metrics(yearly_summary)
    print(f"Avg Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # 3. 收益集中度分析
    analyze_concentration(yearly_stock_pnl)
    
    if not yearly_summary.empty:
        print("\nYEARLY SUMMARY:")
        print(yearly_summary[['year', 'return', 'turnover', 'ic_mean', 'top10_mean_ret']].to_string(index=False))

    results.to_csv('backtest_results.csv', index=False)
    yearly_summary.to_csv('backtest_yearly_summary.csv', index=False)
    plot_results(results)


if __name__ == "__main__":
    main()
