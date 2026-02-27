# AI4Stock: 稳健型 A股机器学习量化框架

这是一个基于 LightGBM 的 A股日频量化回测与训练框架。经过深度重构与优化，本项目特别针对**数据防泄漏（Look-ahead Bias）**、**极端行情（妖股）污染**、**特征分布漂移**以及**网络反爬**等实盘痛点进行了工业级处理。

## 🚀 核心特性 (Features)

### 1. 极致稳健的数据获取 (`fetch_data_manager.py`)
*   **双源聚合**：统一了东方财富（高精准）与新浪财经（高速度）接口，数据结构 100% 互通，无缝切换。
*   **极速体检 (`--check`)**：基于 `pyarrow` 底层元数据读取，**秒级（< 2秒）扫描全市场 5000+ Parquet 文件**，精准定位缺失、断更、含 NaN 的坏文件。
*   **智能增量更新**：读取本地最后日期，仅下载缺失片段；对于新股，自动提取实际上市日并写入本地高速缓存 (`symbols_cache.parquet`)。
*   **工业级反爬与熔断**：集成 `akshare-proxy-patch`（支持 `.env` Token 注入），配置了初始网络探针、单标的超时退出以及连续失败 30 次自动熔断机制，保护账号积分。

### 2. 抗污染的特征工程 (`gen_feature.py`)
*   **丰富的因子库**：抛弃单一动量，引入 RSI(14), ATR(14), MACD(归一化), 均线乖离率 (MA20/60/120 Ratio), 资金流向等稳健技术指标。
*   **风险调整后标签 (Sharpe Proxy Label)**：**这是框架的核心灵魂。** 模型的预测目标不再是绝对收益 (`future_ret_20`)，而是 `future_ret_20 / future_std_20`。强迫模型寻找**稳健上涨**的优质股，而非暴涨暴跌的彩票股。

### 3. 免疫“妖股”的训练引擎 (`train_lgbm.py`)
*   **严格防泄漏**：增加 `GAP_DAYS = 30` 隔离带，确保验证集和训练集的 Label 绝对属于“历史已发生”数据。
*   **截面排名消除漂移 (Cross-Sectional Rank)**：将所有特征转化为每日的 `0~1` 百分位排名，无视 2007年或 2015年的极端市场波动率绝对值，解决跨周期特征分布漂移问题。
*   **动态剔除极端值 (Outlier Rejection)**：每天在构建截面时，**自动丢弃过去 20 天波动率排名前 1% 的股票**。让模型彻底“无视”如 605255 这类依靠资金博弈、连拉一字板的妖股，防止训练权重被扭曲。

### 4. 严苛的回测分析 (`backtest_lgbm.py`)
*   **真实流动性约束 (Limit-Locked Check)**：严密检测一字板（`High == Low`）。如果目标股一字涨停，系统**拒绝买入**；如果一字跌停，系统**拒绝卖出**，彻底消除回测中的“流动性幻觉”。
*   **收益集中度审查 (PnL Concentration)**：按年拆解收益，输出 Top 1 和 Top 10 股票的 PnL 占比。一眼识破策略是真有阿尔法，还是单纯靠踩中了一两只翻倍股。
*   **多维诊断报告**：
    *   **独立年度结算**：每年初重置资金至 100 万，客观评估策略在不同牛熊市下的裸表现。
    *   **特征漂移检测 (KS Test)**：扫描当年数据与全样本的分布差异。
    *   **IC/ICIR 时序热力图**：生成月度预测能力走势图 (`backtest_ic_monthly.png`)。

---

## 🛠️ 快速开始 (Quick Start)

### 0. 环境准备
本项目基于 Python 3.12+ 构建，使用 `uv` 管理依赖。
如果你需要使用东方财富接口，请在根目录创建 `.env` 文件并填入 Proxy Token：
```env
AKSHARE_PROXY_TOKEN=你的Token
```

### 1. 数据下载与体检
```bash
# 检查本地数据健康状况 (极速)
python fetch_data_manager.py --check

# 使用新浪接口快速增量补全
python fetch_data_manager.py --sina

# 使用东财接口高质量全量/增量抓取 (推荐使用 --no-proxy 或配置极慢并发)
python fetch_data_manager.py --hist --workers 1 --sleep 1.0
```

### 2. 生成特征与风险标签
```bash
# 强制覆盖旧特征，生成包含 RSI/ATR 及 Sharpe Label 的数据集
python gen_feature.py --overwrite
```

### 3. 模型训练
```bash
# 清理旧模型（如果在修改特征后）
rm models_lgbm/*.pkl

# 滚动训练过去5年的 LightGBM 模型
python train_lgbm.py
```

### 4. 策略回测
```bash
# 执行带有一字板过滤和收益归因的高级回测
python backtest_lgbm.py
```
*(结果将输出到控制台，并生成 `.csv` 报表及 `.png` 图表)*
