"""app.py - Streamlit page on top of crypto.py (completes the STUDENT TODO).
Run with:  streamlit run app.py
Educational material, not financial advice. Paper trading only - no real orders."""
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import crypto

st.set_page_config(page_title="Crypto Signal Engine", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- 1. TITLE + SIDEBAR
st.title("📈 Crypto Signal Engine")
st.caption("Supervised learning (Decision Tree, Random Forest, XGBoost) → BUY / HOLD / SELL. "
           "Paper trading only. Educational material, not financial advice.")

with st.sidebar:
    st.header("Settings")
    sec = st.slider("Bar size (seconds)", 1, 60, 1)
    ahead = st.slider("Predict ahead (bars)", 1, 60, 10)
    tail_pct = st.slider("BUY / SELL share (% of data each)", 5, 40, 25)
    max_rows = st.select_slider("Max trades to read", options=[50_000, 100_000, 250_000, 500_000], value=250_000)
    st.divider()
    cash = st.number_input("Paper cash (USDT)", 100.0, 1_000_000.0, 1000.0, step=100.0)
    risk = st.slider("Risk per trade (%)", 1, 50, 10)
    st.info("Never time-shuffled: the model trains on the past and is tested on the newest 30%.")


# ---------------------------------------------------------------- 2. DATA
def demo_trades(n=60_000, seed=7):
    """Synthetic trades in the app's own CSV format, so the app works with no file."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2026-10-11 00:00:00") + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 3600, n)), unit="s")
    price = 62000 * np.exp(np.cumsum(rng.normal(0, 4e-5, n)))
    return pd.DataFrame({"timestamp": t, "price": price, "quantity": rng.exponential(0.05, n).round(5) + 0.0001,
                         "side": rng.choice(["buy", "sell"], n)})


@st.cache_data(show_spinner="Reading trades...")
def build_bars(raw_bytes, source, sec, max_rows):
    if source == "demo":
        d = demo_trades()
    else:
        d = crypto.load_trades(io.BytesIO(raw_bytes), nrows=max_rows)
    return crypto.features(crypto.to_bars(d, sec))


uploaded = st.file_uploader("Upload a Binance daily trades CSV (or your own CSV with timestamp, price, quantity, side)",
                            type=["csv"])
use_demo = st.checkbox("Use synthetic demo data instead", value=uploaded is None and not os.path.exists(crypto.DATA_FILE))

bars = None
try:
    if use_demo:
        bars = build_bars(b"", "demo", sec, max_rows)
        st.caption("Showing synthetic demo data.")
    elif uploaded is not None:
        bars = build_bars(uploaded.getvalue(), "upload", sec, max_rows)
    elif os.path.exists(crypto.DATA_FILE):
        with open(crypto.DATA_FILE, "rb") as f:
            bars = build_bars(f.read(), "file", sec, max_rows)
        st.caption(f"Using local file {crypto.DATA_FILE}")
except Exception as e:
    st.error(f"Could not read the file: {e}")

if bars is None or bars.empty:
    st.warning("Upload a trades CSV or tick the demo-data box to continue.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Bars", f"{len(bars):,}")
c2.metric("Last price", f"{bars['price'].iloc[-1]:,.2f}")
c3.metric("Buy flow (last 5 bars)", f"{bars['flow_5'].iloc[-1]:+.2f}" if pd.notna(bars['flow_5'].iloc[-1]) else "n/a")
st.subheader("Price")
st.line_chart(bars.set_index("timestamp")["price"])

# ---------------------------------------------------------------- 3. TRAIN
st.subheader("Train models")
if "trained" not in st.session_state:
    st.session_state.trained = {}

if st.button("Train models", type="primary"):
    rows, fitted = [], {}
    with st.spinner("Training..."):
        for name, model in crypto.get_models().items():
            try:
                acc, base = crypto.train(bars.tail(30_000), model, ahead=ahead, tail=tail_pct / 100)
                rows.append({"Model": name, "Accuracy": round(acc, 3), "Baseline": round(base, 3),
                             "Beats baseline": "Yes" if acc > base else "No"})
                fitted[name] = model
            except Exception as e:
                rows.append({"Model": name, "Accuracy": None, "Baseline": None, "Beats baseline": f"Error: {e}"})
    st.session_state.trained = fitted
    st.session_state.results = pd.DataFrame(rows)

if "results" in st.session_state:
    st.table(st.session_state.results)
    st.caption("Baseline = always guessing the most common class on the test set. "
               "A model that does not beat it has no edge.")

# ---------------------------------------------------------------- 4. LIVE SIGNAL
st.subheader("Live signal")
if st.session_state.trained:
    choice = st.selectbox("Model", list(st.session_state.trained))
    if st.button("Live signal"):
        try:
            sig, conf, price = crypto.live_signal(st.session_state.trained[choice], sec)
            a, b = st.columns(2)
            a.metric("Signal", sig)
            b.metric("Confidence", f"{conf:.0%}", help=f"Price {price:,.2f} {crypto.SYMBOL}")
            st.info(crypto.paper_plan(sig, price, cash, risk))
        except Exception as e:
            st.error(f"Live data unavailable ({e}). Check your internet connection and BINANCE_BASE in .env.")
else:
    st.write("Train the models first.")
