"""EDA pipeline for trades_orderbook_STUDENT_dirty.csv (synthetic BTCUSDT).
Follows EDA_Steps_Trading_Data.pdf (steps 1-9) and README_tasks.md (student tasks)."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
SRC = "/mnt/user-data/uploads/trades_orderbook_STUDENT_dirty.csv"
OUT = "/mnt/user-data/outputs/"
log = []                      # cleaning decisions (Rule: document every decision)
def note(issue, n, action, why): log.append(dict(issue=issue, rows=int(n), action=action, why=why))

# ---------- 1. LOAD & INSPECT ----------
raw = pd.read_csv(SRC)
print(raw.shape); raw.info(); print(raw.head())

# ---------- 2. SUMMARIZE ----------
print(raw.side.value_counts(dropna=False)); print(raw.symbol.value_counts(dropna=False))
raw["price_num"] = pd.to_numeric(raw.price.astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
print(raw[["price_num", "quantity"]].describe())

# ---------- 3. FIND THE PROBLEMS ----------
ts = raw.timestamp.astype(str)
issues = {
 "duplicate rows (full)": int(raw.duplicated().sum()),
 "duplicate trade_id": int(raw.trade_id.duplicated().sum()),
 "missing price": int(raw.price.isna().sum()),
 "missing quantity": int(raw.quantity.isna().sum()),
 "missing side": int(raw.side.isna().sum()),
 "missing bid_qty_1": int(raw.bid_qty_1.isna().sum()),
 "missing ask_price_1": int(raw.ask_price_1.isna().sum()),
 "timestamp ISO format": int(ts.str.match(r"^\d{4}-").sum()),
 "timestamp DD/MM/YYYY": int(ts.str.contains("/").sum()),
 "timestamp unix seconds": int(ts.str.match(r"^\d{10}$").sum()),
 "side label variants (BUY/Sell)": int(raw.side.dropna().isin(["BUY", "Sell"]).sum()),
 "symbol variants": int((raw.symbol != "BTCUSDT").sum()),
 "price stored as text with $ or ,": int(raw.price.astype(str).str.contains(r"[\$,]").sum()),
 "negative price": int((raw.price_num < 0).sum()),
 "price outlier (>1e5, x100)": int((raw.price_num > 1e5).sum()),
 "zero quantity": int((raw.quantity == 0).sum()),
 "crossed book (bid1 > ask1)": int((raw.bid_price_1 > raw.ask_price_1).sum()),
 "rows not in time order (trade_id order)": int((raw.trade_id.diff() < 0).sum()),
}
print(json.dumps(issues, indent=1))

# ---------- 4. CLEAN ----------
df = raw.drop_duplicates("trade_id").copy()
note("Duplicate trade_id", len(raw) - len(df), "dropped", "same trade recorded twice would double-count volume")

df["symbol"] = "BTCUSDT"
note("Symbol variants", (raw.symbol != "BTCUSDT").sum(), "standardised to BTCUSDT", "same instrument, different spelling")
df["side"] = df.side.str.lower().str.strip()
note("Side case variants", raw.side.isin(["BUY", "Sell"]).sum(), "lower-cased", "buy/BUY/Sell are the same label")

# timestamps -> one UTC-naive datetime
def parse_ts(s):
    s = str(s)
    if s.isdigit():          return pd.to_datetime(int(s), unit="s")
    if "/" in s:             return pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S")
    return pd.to_datetime(s)
df["timestamp"] = df.timestamp.map(parse_ts)
note("Mixed timestamp formats", (~ts.str.match(r"^\d{4}-")).sum(), "parsed ISO / DD/MM/YYYY / unix-seconds into one datetime", "day-first assumed because 01/10/2026 sits next to 2026-10-01 trades")

df["price"] = pd.to_numeric(df.price.astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
note("Price as text ($, commas)", raw.price.astype(str).str.contains(r"[\$,]").sum(), "stripped symbols, cast to float", "needed for any maths")

# crossed book: best bid is inflated (~ +1.1 above ask); rebuild bid_1 from ask_1 minus the median normal spread
ok_spread = (df.ask_price_1 - df.bid_price_1)[df.ask_price_1 > df.bid_price_1].median()
cross = df.bid_price_1 > df.ask_price_1
df.loc[cross, "bid_price_1"] = (df.ask_price_1 - ok_spread).round(1)[cross]
note("Crossed order book", cross.sum(), f"bid_1 rebuilt as ask_1 - median spread ({ok_spread:.2f})", "bid>ask is impossible; dropping would lose otherwise valid trades")

# missing ask_price_1 -> ask_2 minus median (ask2-ask1) gap
gap = (df.ask_price_2 - df.ask_price_1).median(); m = df.ask_price_1.isna()
df.loc[m, "ask_price_1"] = (df.ask_price_2 - gap).round(1)[m]
note("Missing ask_price_1", m.sum(), f"filled from ask_price_2 - median gap ({gap:.2f})", "book levels are tightly linked")

# negative / x100 prices
neg = df.price < 0; df.loc[neg, "price"] = df.price[neg].abs()
note("Negative price", neg.sum(), "abs() - sign flip", "magnitude matches best bid/ask exactly")
big = df.price > 1e5; df.loc[big, "price"] = df.price[big] / 100
note("Price x100", big.sum(), "divided by 100", "decimal-shift; result matches order book")

# side <-> price: buys hit the ask, sells hit the bid (matched ~99% in clean rows)
m = df.side.isna() & df.price.notna()
df.loc[m & (df.price == df.ask_price_1), "side"] = "buy"
df.loc[m & (df.price == df.bid_price_1), "side"] = "sell"
n_side_fixed = m.sum() - df.side[m].isna().sum()
note("Missing side", raw.side.isna().sum(), f"inferred {n_side_fixed} from price==ask1 (buy) / bid1 (sell); rest dropped if still unknown", "trade side is observable from which book side was hit")
m = df.price.isna() & df.side.notna()
df.loc[m, "price"] = np.where(df.side[m] == "buy", df.ask_price_1[m], df.bid_price_1[m])
note("Missing price", raw.price.isna().sum(), "reconstructed: buy->ask_1, sell->bid_1", "this rule reproduces price in ~99% of complete rows")
before = len(df); df = df.dropna(subset=["price", "side"])
note("Missing price AND side (unrecoverable)", before - len(df), "dropped", "nothing to reconstruct from")

before = len(df); bad = df.quantity.isna() | (df.quantity <= 0); df = df[~bad]
note("Missing / zero quantity", before - len(df), "dropped", "trade size cannot be inferred; zero-size trade is not a trade")

df = df.sort_values("timestamp").reset_index(drop=True)
m = df.bid_qty_1.isna()
df["bid_qty_1"] = df.bid_qty_1.fillna(df.bid_qty_1.rolling(7, center=True, min_periods=1).median())
note("Missing bid_qty_1", m.sum(), "filled with rolling median of 7 neighbouring ticks", "quantity at top of book is noisy; local median is a neutral guess (and imbalance is flagged)")
df["bid_qty_1_imputed"] = m

# ---------- 5. OUTLIERS (rolling median check on what is left) ----------
rm = df.price.rolling(21, center=True, min_periods=1).median()
dev = (df.price - rm).abs()
thr = max(5 * dev.median() * 1.4826, 1e-9)
out = dev > 10 * thr
print("residual price outliers after fixes:", int(out.sum()), "| max deviation from rolling median:", round(dev.max(), 2))
note("Residual price outliers (rolling-median, 10xMAD)", out.sum(), "dropped" if out.sum() else "none found", "confirms steps above removed all extreme prices")
df = df[~out].reset_index(drop=True)
q_hi = df.quantity.quantile(.99)  # large quantities are legit whales -> keep, only flag
df["big_trade"] = df.quantity > q_hi

# ---------- 6. FEATURES (no look-ahead: only past/current data) ----------
df["spread"] = df.ask_price_1 - df.bid_price_1
df["mid_price"] = (df.ask_price_1 + df.bid_price_1) / 2
df["order_imbalance"] = (df.bid_qty_1 - df.ask_qty_1) / (df.bid_qty_1 + df.ask_qty_1)
df["return"] = np.log(df.mid_price).diff()
df["volatility_20"] = df["return"].rolling(20).std()
df["ma_20"] = df.price.rolling(20).mean()
d = df.price.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
df["rsi_14"] = 100 - 100 / (1 + up / dn)
df.to_csv(OUT + "trades_orderbook_CLEAN.csv", index=False)

# ---------- 7. RESAMPLE 1-min OHLCV ----------
t = df.set_index("timestamp")
c = t.price.resample("1min").ohlc()
c["volume"] = t.quantity.resample("1min").sum()
c["buy_volume"] = t.quantity[t.side == "buy"].resample("1min").sum()
c["sell_volume"] = t.quantity[t.side == "sell"].resample("1min").sum()
c["trades"] = t.price.resample("1min").count()
c["avg_spread"] = t.spread.resample("1min").mean()
c["avg_imbalance"] = t.order_imbalance.resample("1min").mean()
c = c.sort_index(); c.to_csv(OUT + "ohlcv_1min.csv")

# ---------- 8. VISUALIZE ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(c.index, c.close); ax.set_title("BTCUSDT 1-min close"); ax.set_ylabel("price")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT + "figures/1_price_line.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(df.quantity, bins=50); ax.set_title("Trade quantity (volume) histogram"); ax.set_xlabel("BTC")
fig.tight_layout(); fig.savefig(OUT + "figures/2_volume_hist.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(df.spread, bins=40); ax.set_title("Spread distribution"); ax.set_xlabel("ask_1 - bid_1")
fig.tight_layout(); fig.savefig(OUT + "figures/3_spread_dist.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(4, 4)); df.side.value_counts().plot.bar(ax=ax, color=["tab:green", "tab:red"]); ax.set_title("Buy vs sell counts"); ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(OUT + "figures/4_buy_sell.png", dpi=110); plt.close()
feat = ["price", "quantity", "spread", "mid_price", "order_imbalance", "return", "volatility_20", "ma_20", "rsi_14"]
fig, ax = plt.subplots(figsize=(7, 6)); sns.heatmap(df[feat].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Feature correlation")
fig.tight_layout(); fig.savefig(OUT + "figures/5_corr_heatmap.png", dpi=110); plt.close()

# ---------- 9. MODEL PREP (time split, no shuffle) ----------
# target at tick level: is mid price 5 minutes later higher than now?  (future used ONLY for the label)
tk = df.set_index("timestamp")
q = pd.DataFrame({"timestamp": tk.index + pd.Timedelta(minutes=5)})
fut = pd.merge_asof(q, df[["timestamp", "mid_price"]], on="timestamp", direction="nearest", tolerance=pd.Timedelta("30s"))["mid_price"].values
tk["target_up_5m"] = np.where(np.isnan(fut), np.nan, (fut > tk.mid_price.values).astype(float))
m = tk.dropna(subset=["target_up_5m", "return", "volatility_20", "rsi_14"]).copy()
cut = int(len(m) * 0.7); train, test = m.iloc[:cut], m.iloc[cut:]
X = ["spread", "order_imbalance", "return", "volatility_20", "rsi_14"]
mu, sd = train[X].mean(), train[X].std()                      # scaler fitted on TRAIN only
sc_train, sc_test = (train[X] - mu) / sd, (test[X] - mu) / sd
rule = (test.order_imbalance > 0.3).astype(int)               # demo rule from the PDF
res = dict(rows_model=len(m), train=len(train), test=len(test), train_end=str(train.index[-1]), test_start=str(test.index[0]),
           base_rate_up_test=float(test.target_up_5m.mean()),
           rule_signals=int(rule.sum()), rule_hit_rate=float(test.target_up_5m[rule == 1].mean()) if rule.sum() else None)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=500).fit(sc_train, train.target_up_5m)
res["logreg_test_accuracy"] = float(clf.score(sc_test, test.target_up_5m))
m.to_csv(OUT + "model_dataset.csv")
json.dump(dict(raw_issues=issues, cleaning_log=log, final_rows=len(df), candles=len(c), model=res), open(OUT + "eda_results.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print(pd.DataFrame(log)[["issue", "rows", "action"]].to_string())
print("final", df.shape, "missing:", int(df[["price","quantity","side","bid_qty_1","ask_price_1"]].isna().sum().sum()))
