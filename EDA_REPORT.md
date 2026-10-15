# EDA Report: trades_orderbook_STUDENT_dirty.csv (synthetic BTCUSDT)

Educational material, not financial advice. Data is synthetic.

**Raw:** 2,035 rows x 18 cols. **Clean:** 1,974 rows. **1-min candles:** 28.

## Task 1: Data quality issues found

| Issue | Count |
|---|---|
| duplicate rows (full) | 35 |
| duplicate trade_id | 35 |
| missing price | 25 |
| missing quantity | 20 |
| missing side | 15 |
| missing bid_qty_1 | 30 |
| missing ask_price_1 | 20 |
| timestamp ISO format | 1917 |
| timestamp DD/MM/YYYY | 78 |
| timestamp unix seconds | 40 |
| side label variants (BUY/Sell) | 120 |
| symbol variants | 61 |
| price stored as text with $ or , | 50 |
| negative price | 6 |
| price outlier (>1e5, x100) | 8 |
| zero quantity | 6 |
| crossed book (bid1 > ask1) | 10 |
| rows not in time order (trade_id order) | 1013 |

## Tasks 2-3: Cleaning decisions (every one documented)

| Issue | Rows | Action | Why |
|---|---|---|---|
| Duplicate trade_id | 35 | dropped | same trade recorded twice would double-count volume |
| Symbol variants | 61 | standardised to BTCUSDT | same instrument, different spelling |
| Side case variants | 120 | lower-cased | buy/BUY/Sell are the same label |
| Mixed timestamp formats | 118 | parsed ISO / DD/MM/YYYY / unix-seconds into one datetime | day-first assumed because 01/10/2026 sits next to 2026-10-01 trades |
| Price as text ($, commas) | 50 | stripped symbols, cast to float | needed for any maths |
| Crossed order book | 10 | bid_1 rebuilt as ask_1 - median spread (0.10) | bid>ask is impossible; dropping would lose otherwise valid trades |
| Missing ask_price_1 | 20 | filled from ask_price_2 - median gap (0.20) | book levels are tightly linked |
| Negative price | 6 | abs() - sign flip | magnitude matches best bid/ask exactly |
| Price x100 | 8 | divided by 100 | decimal-shift; result matches order book |
| Missing side | 15 | inferred 15 from price==ask1 (buy) / bid1 (sell); rest dropped if still unknown | trade side is observable from which book side was hit |
| Missing price | 25 | reconstructed: buy->ask_1, sell->bid_1 | this rule reproduces price in ~99% of complete rows |
| Missing price AND side (unrecoverable) | 0 | dropped | nothing to reconstruct from |
| Missing / zero quantity | 26 | dropped | trade size cannot be inferred; zero-size trade is not a trade |
| Missing bid_qty_1 | 30 | filled with rolling median of 7 neighbouring ticks | quantity at top of book is noisy; local median is a neutral guess (and imbalance is flagged) |
| Residual price outliers (rolling-median, 10xMAD) | 0 | none found | confirms steps above removed all extreme prices |

Rows were then sorted by time. 2,000 unique trades minus 26 with missing/zero quantity = 1,974 rows. Zero missing values remain in key columns.

## Task 4: Features
`spread`, `mid_price`, `order_imbalance`, plus log return, 20-tick rolling volatility, 20-tick MA, RSI(14). All use only past/current data (no look-ahead).

## Task 5: 1-minute OHLCV
`ohlcv_1min.csv` with open/high/low/close, volume, buy/sell volume, trade count, avg spread, avg imbalance.

## Charts
See `figures/` (price line, volume histogram, spread distribution, buy vs sell, correlation heatmap).

## Step 9: Model prep

- Target: mid price higher 5 minutes later (label is the only place future data is used).
- Time split (no shuffle): train 1129 rows (to 2026-10-01 09:15:40.080000), test 484 rows (from 2026-10-01 09:15:40.132000). Scaler fitted on train only.
- Base rate of 'up' in test: 71.9%.
- Rule `order_imbalance > 0.3`: 167 signals, hit rate 70.7% (no better than the base rate, so no edge).
- Logistic regression test accuracy 71.9%, identical to always predicting 'up'. Data is a short synthetic window, so treat as a demo of the workflow, not a tradable signal.
- Next per PDF: backtest, then paper trading via 3Commas webhook.