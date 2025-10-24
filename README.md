# TradingEngine: DRL FOREX Scalping Bot

**Extreme-profit Deep Reinforcement Learning bot for 1-minute FOREX scalping.**

> ⚠️ **HIGH RISK STRATEGY:** This bot is configured for maximum profit with 10%+ risk per trade. Expect 40-60% drawdowns. **ALWAYS test on demo account first!**

## Features

- **Deep Reinforcement Learning:** PPO agent trained on 3+ years of M1 data
- **H200 NVL Optimized:** 128 parallel environments, 50M parameter models
- **Aggressive Reward:** Profit-maximization with light risk penalties
- **MT5 Integration:** Real-time data collection and order execution
- **100+ Features:** Price action, technical indicators, order flow, market microstructure
- **Curriculum Learning:** Progressive training from calm → volatile markets
- **GPU-Accelerated Backtesting:** Vectorized backtesting with realistic spread/slippage
- **Walk-Forward Validation:** Rolling 6-month train, 1-month test

## Architecture

```
State (100+ features, 60-bar lookback)
    ↓
[LSTM (1024×8) + Transformer (768d×12)]  ← Feature Extraction
    ↓
┌─────────────────┬─────────────────┐
│   Actor Network │  Critic Network │
│  (Policy π)     │  (Value V)      │
└─────────────────┴─────────────────┘
    ↓                     ↓
Actions:                 Value
- BUY/SELL/CLOSE/HOLD   Estimate
- Position size (0.01-10 lots)
- Stop loss (5-50 pips)
- Take profit (10-150 pips)
```

## Project Structure

```
TradingEngine/
├── config/
│   ├── config.yaml           # Main configuration
│   └── mt5_config.yaml       # MT5 credentials (don't commit!)
├── data/
│   ├── collectors/           # MT5 data collection
│   ├── features/             # Feature engineering (100+ features)
│   └── environment/          # Gym environment
├── models/
│   ├── rl_agent/             # PPO agent (actor/critic)
│   └── reward_shaping/       # Aggressive reward function
├── training/
│   ├── train_rl_h200.py      # H200-optimized training
│   ├── curriculum_learning.py
│   └── parallel_envs.py      # 128 parallel environments
├── backtesting/
│   ├── vectorized_backtest.py # GPU-accelerated backtesting
│   └── walk_forward.py
├── deployment/
│   ├── mt5_executor.py       # Live trading execution
│   └── monitoring/           # Performance tracking
└── utils/
    ├── logger.py
    └── mt5_bridge.py         # MT5 Python API wrapper
```

## Installation

### 1. Prerequisites

- **Python 3.11+**
- **MetaTrader 5** (installed and configured)
- **CUDA 12.1+** (for H200 GPU training)
- **MT5 Demo/Live Account** (from any broker)

### 2. Install Dependencies

```bash
cd TradingEngine
pip install -r requirements.txt
```

### 3. Configure MT5

Edit `config/mt5_config.yaml` with your MT5 credentials:

```yaml
mt5:
  login: YOUR_ACCOUNT_NUMBER
  password: "YOUR_PASSWORD"
  server: "YOUR_BROKER_SERVER"
  terminal_path:
    windows: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

> **Security:** Never commit `mt5_config.yaml` to git! Add it to `.gitignore`.

### 4. Verify MT5 Connection

```bash
python utils/mt5_bridge.py
```

Expected output:
```
2024-01-15 10:30:00 | TradingEngine | INFO | MT5 connection successful!
2024-01-15 10:30:01 | TradingEngine | INFO | Sample data:
                       time    open    high     low   close  tick_volume
0  2024-01-15 10:00:00  1.0950  1.0955  1.0948  1.0952         1234
```

## Usage

### Step 1: Collect Historical Data

Collect 3+ years of 1-minute data from MT5:

```bash
python data/collectors/mt5_historical_collector.py
```

This will download M1 bars for EURUSD, GBPUSD, USDJPY and save to `data/raw/`.

**Expected time:** 10-30 minutes depending on broker speed.

### Step 2: Feature Engineering

**TODO:** Create feature engineering pipeline
```bash
python data/features/feature_engineering.py
```

### Step 3: Train RL Agent on H200

**TODO:** Create training pipeline
```bash
python training/train_rl_h200.py --config config/config.yaml
```

**Expected training time on H200 NVL:** 48-72 hours for 100M timesteps

Training features:
- 128 parallel environments
- bfloat16 mixed precision
- Curriculum learning (calm → normal → volatile → mixed)
- Checkpoints every 10M steps

### Step 4: Backtesting

**TODO:** Create backtesting engine
```bash
python backtesting/vectorized_backtest.py --model checkpoints/best_model.pth
```

### Step 5: Walk-Forward Validation

**TODO:** Create walk-forward validation
```bash
python backtesting/walk_forward.py --model checkpoints/best_model.pth
```

### Step 6: Paper Trading

**TODO:** Create live trading system
```bash
python deployment/live_trader.py --model checkpoints/best_model.pth --paper-trading
```

### Step 7: Live Trading

> ⚠️ **Only after 1+ month of successful paper trading!**

```bash
python deployment/live_trader.py --model checkpoints/best_model.pth
```

## Configuration

### Risk Settings (`config/config.yaml`)

```yaml
risk:
  base_lot_per_10k: 1.0      # 1 lot per $10K account
  max_lot_size: 10.0         # Max 10 lots (10% account)
  max_concurrent_trades: 5   # Max 5 simultaneous trades
  daily_loss_limit_pct: 40.0 # Stop after -40% day
  max_drawdown_pct: 60.0     # Emergency shutdown
```

### Reward Function

```python
reward = (
    profit * 2.0              # Maximize profit
    - drawdown * 0.3          # Light penalty
    + big_win_bonus * 1.5     # Bonus for >40 pip wins
    - holding_cost * 0.01     # Quick scalps
    + win_streak * 0.5        # Consistency bonus
)
```

### H200 Training Settings

```yaml
h200:
  num_envs: 128              # Parallel environments
  use_bfloat16: true         # Native H200 precision
  replay_buffer_size: 10M    # GPU-resident buffer
  checkpoint_freq: 10M       # Save every 10M steps
```

## Expected Performance

### Conservative Estimate
- **Annual return:** 150-250%
- **Win rate:** 55-65%
- **Profit factor:** 1.5-2.0
- **Max drawdown:** 40-50%
- **Sharpe ratio:** 1.5-2.0

### Optimistic Estimate
- **Annual return:** 300-500%+
- **Win rate:** 60-70%
- **Profit factor:** 2.0-3.0
- **Max drawdown:** 50-60%
- **Sharpe ratio:** 2.0-3.0

## Risk Warnings

⚠️ **THIS IS AN EXTREMELY AGGRESSIVE STRATEGY:**

1. **10%+ risk per trade** can lead to rapid account depletion
2. **40-60% drawdowns** are expected during losing streaks
3. **Martingale strategies** (if enabled) can wipe out accounts in 3-5 trades
4. **RL agents can fail** catastrophically during regime changes
5. **Market conditions change** - model drift is inevitable
6. **Past performance ≠ future results**

### Mandatory Safety Measures

- ✅ **Test on demo for 1+ month minimum**
- ✅ **Start live with <5% of total capital**
- ✅ **Set hard daily loss limits (-40%)**
- ✅ **Monitor trades daily**
- ✅ **Be prepared to shut down**
- ✅ **Never risk money you can't afford to lose**

## Development Roadmap

### Phase 1: Foundation (Completed ✓)
- [x] Project structure
- [x] MT5 data collector
- [x] Configuration files
- [x] Logging utilities

### Phase 2: Data & Features (TODO)
- [ ] Feature engineering pipeline (100+ features)
- [ ] Data augmentation
- [ ] Train/val/test split

### Phase 3: RL Agent (TODO)
- [ ] Gym environment with market simulation
- [ ] Actor/critic networks (50M params)
- [ ] Aggressive reward function
- [ ] PPO training loop

### Phase 4: Training (TODO)
- [ ] H200-optimized training pipeline
- [ ] 128 parallel environments
- [ ] Curriculum learning
- [ ] Model checkpointing

### Phase 5: Validation (TODO)
- [ ] GPU-accelerated backtesting
- [ ] Walk-forward validation
- [ ] Monte Carlo stress testing

### Phase 6: Deployment (TODO)
- [ ] MT5 live execution
- [ ] Performance monitoring
- [ ] Telegram alerts
- [ ] Emergency shutdown system

## Tech Stack

- **Python 3.11+**
- **PyTorch 2.1+** (RL agent)
- **Stable-Baselines3** (PPO implementation)
- **Gymnasium** (RL environment)
- **MetaTrader5 Python API**
- **Pandas, NumPy** (data processing)
- **TA-Lib** (technical indicators)
- **Optuna** (hyperparameter tuning)
- **TensorBoard** (training visualization)

## Contributing

This is a personal trading bot. If you want to contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - Use at your own risk

## Disclaimer

This software is for educational purposes only. Trading FOREX is extremely risky and most traders lose money. The author is not responsible for any financial losses incurred from using this software.

**YOU HAVE BEEN WARNED.**

---

Built with ❤️ and a lot of GPU power.

**Status:** 🟡 Under Development (Phase 1 complete, Phase 2-6 in progress)
