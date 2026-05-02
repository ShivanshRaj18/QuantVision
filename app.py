# ============================================================
#   QuantVision — AI-Powered Portfolio Optimizer
#   Author: [Your Name] | DTU Mathematics & Computing
#   Stack:  Python · Streamlit · Scikit-Learn · Plotly
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision | AI Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark gradient header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0 0 0;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    /* Style metric boxes */
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 QuantVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Portfolio Optimizer · Built on Modern Portfolio Theory + Machine Learning</div>', unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────
#  SIDEBAR — User Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 📌 Stock Tickers")
    tickers_raw = st.text_input(
        "Enter comma-separated tickers",
        value="AAPL, GOOGL, MSFT, AMZN, TSLA",
        help="E.g.  AAPL, GOOGL, MSFT, TCS.NS (Indian stocks use .NS)"
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    st.markdown("### 📅 Analysis Period")
    start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
    end_date   = st.date_input("End Date",   value=datetime.today())

    st.markdown("### 💰 Investment Amount")
    investment = st.number_input("Total Capital (USD $)", min_value=100, value=10000, step=500)

    st.markdown("### 🏦 Risk-Free Rate")
    rf_rate = st.slider(
        "Annual Risk-Free Rate (%)",
        min_value=0.0, max_value=12.0, value=4.5, step=0.1,
        help="Use current 10-year US treasury yield (~4.5%)"
    ) / 100

    st.markdown("### 🎯 Optimization Goal")
    opt_goal = st.radio(
        "Optimize for:",
        ["Max Sharpe Ratio", "Min Volatility"],
        help="Sharpe = best return per unit of risk. Min Vol = safest portfolio."
    )

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Data: Yahoo Finance · Model: Random Forest · Math: Markowitz MPT")


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    """Download adjusted closing prices from Yahoo Finance."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):           # single ticker edge-case
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(axis=1, how="all").dropna()


# ─────────────────────────────────────────────
#  PORTFOLIO MATH (Markowitz MPT)
# ─────────────────────────────────────────────
def port_stats(w, mu, cov, rf):
    """Return (annual_return, annual_vol, sharpe) for weight vector w."""
    ret = float(np.dot(w, mu)) * 252
    vol = float(np.sqrt(w @ (cov * 252) @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe


def optimize(returns_df, rf, goal="sharpe"):
    """
    Find optimal portfolio weights via constrained numerical optimization.
    Uses scipy.optimize.minimize with SLSQP method.
    """
    n   = len(returns_df.columns)
    mu  = returns_df.mean()
    cov = returns_df.cov()
    w0  = np.full(n, 1 / n)                       # equal-weight starting point
    bounds      = [(0.01, 0.60)] * n               # each asset: 1%–60%
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    if goal == "sharpe":
        def objective(w):
            return -port_stats(w, mu, cov, rf)[2]  # maximise Sharpe ⟹ minimise –Sharpe
    else:
        def objective(w):
            return port_stats(w, mu, cov, rf)[1]   # minimise volatility

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    return result.x


def efficient_frontier_cloud(returns_df, rf, n=4000):
    """Simulate random portfolios to draw the Efficient Frontier scatter cloud."""
    n_assets = len(returns_df.columns)
    mu  = returns_df.mean() * 252
    cov = returns_df.cov()  * 252
    rets, vols, sharpes = [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r = float(np.dot(w, mu))
        v = float(np.sqrt(w @ cov @ w))
        rets.append(r);  vols.append(v);  sharpes.append((r - rf) / v if v else 0)
    return pd.DataFrame({"return": rets, "volatility": vols, "sharpe": sharpes})


# ─────────────────────────────────────────────
#  RISK METRICS
# ─────────────────────────────────────────────
def value_at_risk(ret_series, conf=0.95):
    """Historical VaR at given confidence level."""
    return float(np.percentile(ret_series, (1 - conf) * 100))

def max_drawdown(price_series):
    """Maximum peak-to-trough drawdown."""
    rolling_max = price_series.cummax()
    return float(((price_series - rolling_max) / rolling_max).min())

def beta(stock_ret, bench_ret):
    """Beta relative to benchmark."""
    cov_mat = np.cov(stock_ret, bench_ret)
    return cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 1.0


# ─────────────────────────────────────────────
#  ML PREDICTOR
# ─────────────────────────────────────────────
def build_features(price_s: pd.Series) -> pd.DataFrame:
    """
    Engineer technical-indicator features from a price series.
    Features used:
      - Log returns, MA ratios (5/10/20/50-day), RSI-14,
        realised volatility (5/20-day), momentum (5/10/20-day).
    """
    df = pd.DataFrame(index=price_s.index)
    ret = price_s.pct_change()
    log_ret = np.log(price_s / price_s.shift(1))

    df["ret"]          = ret
    df["log_ret"]      = log_ret
    df["ma5_ratio"]    = price_s.rolling(5).mean()  / price_s - 1
    df["ma10_ratio"]   = price_s.rolling(10).mean() / price_s - 1
    df["ma20_ratio"]   = price_s.rolling(20).mean() / price_s - 1
    df["ma50_ratio"]   = price_s.rolling(50).mean() / price_s - 1
    df["vol5"]         = ret.rolling(5).std()
    df["vol20"]        = ret.rolling(20).std()
    df["mom5"]         = price_s / price_s.shift(5)  - 1
    df["mom10"]        = price_s / price_s.shift(10) - 1
    df["mom20"]        = price_s / price_s.shift(20) - 1

    # RSI-14
    delta = price_s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df["target"] = ret.shift(-1)   # next-day return = what we predict
    return df.dropna()


def train_predictor(price_s: pd.Series):
    """
    Train a Random Forest Regressor.
    Returns: model, scaler, X_test, y_test, y_pred, direction_accuracy
    """
    feats = build_features(price_s)
    X = feats.drop("target", axis=1)
    y = feats["target"]

    split = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestRegressor(n_estimators=150, max_depth=8,
                                  random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)

    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_te)))
    return model, scaler, X_te, y_te, y_pred, dir_acc


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
# Load data immediately (cached) so the page isn't blank on first load
with st.spinner("📡 Fetching market data from Yahoo Finance …"):
    try:
        prices = load_prices(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

valid_tickers = list(prices.columns)
if not valid_tickers:
    st.error("No valid tickers found. Please check your input and try again.")
    st.stop()

daily_ret = prices.pct_change().dropna()
COLORS    = px.colors.qualitative.Plotly

TAB_OVERVIEW, TAB_OPTIMIZER, TAB_RISK, TAB_AI = st.tabs([
    "📈  Market Overview",
    "🎯  Portfolio Optimizer",
    "⚠️  Risk Analysis",
    "🤖  AI Predictor",
])


# ══════════════════════════════════════════════
#  TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════
with TAB_OVERVIEW:
    st.subheader("📈 Live Stock Performance")

    # ── Top metric row ──
    cols = st.columns(len(valid_tickers))
    for i, tkr in enumerate(valid_tickers):
        cur  = prices[tkr].iloc[-1]
        prev = prices[tkr].iloc[-2]
        tot  = (prices[tkr].iloc[-1] / prices[tkr].iloc[0] - 1) * 100
        cols[i].metric(
            label=f"**{tkr}**",
            value=f"${cur:,.2f}",
            delta=f"{(cur/prev-1)*100:.2f}% today  |  {tot:.1f}% total"
        )

    st.divider()

    # ── Normalised price chart ──
    st.subheader("📊 Normalised Price Chart  (Base = 100)")
    norm = prices / prices.iloc[0] * 100
    fig_price = go.Figure()
    for i, tkr in enumerate(valid_tickers):
        fig_price.add_trace(go.Scatter(
            x=norm.index, y=norm[tkr], name=tkr,
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            hovertemplate=f"<b>{tkr}</b>  %{{y:.1f}}<extra></extra>"
        ))
    fig_price.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
    fig_price.update_layout(template="plotly_dark", height=420,
                            xaxis_title="Date", yaxis_title="Normalised Price",
                            legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Returns distribution + Correlation ──
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📉 Daily Returns Distribution")
        fig_hist = go.Figure()
        for i, tkr in enumerate(valid_tickers):
            fig_hist.add_trace(go.Histogram(
                x=daily_ret[tkr], name=tkr, nbinsx=60, opacity=0.65
            ))
        fig_hist.update_layout(barmode="overlay", template="plotly_dark", height=360,
                               xaxis_title="Daily Return", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("🔥 Correlation Heatmap")
        corr = daily_ret.corr()
        fig_corr = px.imshow(corr, text_auto=".2f",
                             color_continuous_scale="RdYlGn",
                             zmin=-1, zmax=1)
        fig_corr.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Summary stats table ──
    st.subheader("📋 Summary Statistics")
    stats = pd.DataFrame({
        "Ticker": valid_tickers,
        "Total Return": [f"{(prices[t].iloc[-1]/prices[t].iloc[0]-1)*100:.2f}%" for t in valid_tickers],
        "Ann. Return":  [f"{daily_ret[t].mean()*252*100:.2f}%"                  for t in valid_tickers],
        "Ann. Volatility": [f"{daily_ret[t].std()*np.sqrt(252)*100:.2f}%"       for t in valid_tickers],
        "Best Day":     [f"{daily_ret[t].max()*100:.2f}%"                        for t in valid_tickers],
        "Worst Day":    [f"{daily_ret[t].min()*100:.2f}%"                        for t in valid_tickers],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 2 — PORTFOLIO OPTIMIZER
# ══════════════════════════════════════════════
with TAB_OPTIMIZER:
    st.subheader("🎯 Markowitz Modern Portfolio Theory Optimizer")

    st.info(
        "**What this does:** Uses the Nobel Prize-winning Markowitz Mean-Variance Optimization "
        "to find the mathematically best way to split your \\$" + f"{investment:,.0f} "
        "across the selected stocks — maximising return for the risk you take."
    )

    with st.spinner("🧮 Running optimisation…"):
        goal_key    = "sharpe" if opt_goal == "Max Sharpe Ratio" else "min_vol"
        w_opt       = optimize(daily_ret, rf_rate, goal_key)
        mu          = daily_ret.mean()
        cov         = daily_ret.cov()
        r_opt, v_opt, s_opt = port_stats(w_opt, mu, cov, rf_rate)
        ef          = efficient_frontier_cloud(daily_ret, rf_rate)

    # ── Key metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📈 Expected Annual Return",  f"{r_opt*100:.2f}%")
    m2.metric("📉 Annual Volatility (Risk)", f"{v_opt*100:.2f}%")
    m3.metric("⭐ Sharpe Ratio",              f"{s_opt:.3f}")
    m4.metric("💰 Expected 1-yr Gain",       f"${investment*r_opt:,.0f}")

    st.divider()
    c1, c2 = st.columns([1.3, 0.7])

    with c1:
        # Efficient Frontier scatter
        st.subheader("🌊 Efficient Frontier")
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=ef["volatility"], y=ef["return"], mode="markers",
            marker=dict(size=3, color=ef["sharpe"],
                        colorscale="Viridis", opacity=0.55,
                        colorbar=dict(title="Sharpe")),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
        ))
        fig_ef.add_trace(go.Scatter(
            x=[v_opt], y=[r_opt], mode="markers",
            marker=dict(size=18, color="red", symbol="star"),
            name="⭐ Optimal Portfolio",
            hovertemplate=(f"<b>OPTIMAL</b><br>Vol: {v_opt:.2%}"
                           f"<br>Ret: {r_opt:.2%}<br>Sharpe: {s_opt:.3f}<extra></extra>")
        ))
        fig_ef.update_layout(
            template="plotly_dark", height=430,
            xaxis=dict(title="Annual Volatility", tickformat=".1%"),
            yaxis=dict(title="Annual Return",     tickformat=".1%"),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    with c2:
        st.subheader("🥧 Optimal Weights")
        fig_pie = px.pie(
            values=w_opt, names=valid_tickers,
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(template="plotly_dark", height=430,
                              showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Allocation table ──
    st.subheader("💼 Detailed Allocation")
    alloc = pd.DataFrame({
        "Stock":               valid_tickers,
        "Weight":              [f"{w*100:.1f}%"          for w in w_opt],
        "Amount Invested":     [f"${w*investment:,.2f}"  for w in w_opt],
        "Annual Return Contrib.":[f"{w * daily_ret[t].mean() * 252 * 100:.2f}%"
                                  for w, t in zip(w_opt, valid_tickers)],
    })
    st.dataframe(alloc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 3 — RISK ANALYSIS
# ══════════════════════════════════════════════
with TAB_RISK:
    st.subheader("⚠️ Comprehensive Risk Dashboard")

    # ── Fetch S&P 500 benchmark ──
    with st.spinner("📡 Loading S&P 500 benchmark (SPY)…"):
        try:
            bench_prices = load_prices(["SPY"], start_date, end_date)
            bench_ret    = bench_prices.pct_change().dropna()["SPY"]
            has_bench    = True
        except Exception:
            has_bench = False

    # ── Per-stock risk table ──
    rows = []
    for tkr in valid_tickers:
        r   = daily_ret[tkr]
        p   = prices[tkr]
        ann_r   = r.mean() * 252
        ann_v   = r.std()  * np.sqrt(252)
        sharpe  = (ann_r - rf_rate) / ann_v if ann_v else 0
        var95   = value_at_risk(r, 0.95)
        var99   = value_at_risk(r, 0.99)
        mdd     = max_drawdown(p)
        b       = beta(r.values, bench_ret.values) if has_bench else None

        row = {
            "Ticker":          tkr,
            "Ann. Return":     f"{ann_r*100:.2f}%",
            "Ann. Volatility": f"{ann_v*100:.2f}%",
            "Sharpe Ratio":    f"{sharpe:.3f}",
            "VaR 95%":         f"{var95*100:.2f}%",
            "VaR 99%":         f"{var99*100:.2f}%",
            "Max Drawdown":    f"{mdd*100:.2f}%",
        }
        if has_bench:
            row["Beta (vs SPY)"] = f"{b:.3f}"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📉 Rolling 30-Day Volatility")
        fig_vol = go.Figure()
        for i, tkr in enumerate(valid_tickers):
            rv = daily_ret[tkr].rolling(30).std() * np.sqrt(252) * 100
            fig_vol.add_trace(go.Scatter(
                x=rv.index, y=rv, name=tkr,
                line=dict(color=COLORS[i % len(COLORS)], width=2)
            ))
        fig_vol.update_layout(template="plotly_dark", height=360,
                              yaxis_title="Annualised Vol (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        st.subheader("📊 Drawdown Chart")
        fig_dd = go.Figure()
        for i, tkr in enumerate(valid_tickers):
            dd = (prices[tkr] - prices[tkr].cummax()) / prices[tkr].cummax() * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, name=tkr, fill="tozeroy", opacity=0.45,
                line=dict(color=COLORS[i % len(COLORS)])
            ))
        fig_dd.update_layout(template="plotly_dark", height=360,
                             yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Portfolio-level VaR ──
    st.subheader(f"💼 Portfolio Value-at-Risk  (Investment = ${investment:,.0f})")
    port_r = daily_ret[valid_tickers].dot(w_opt)
    v95 = abs(value_at_risk(port_r, 0.95)) * investment
    v99 = abs(value_at_risk(port_r, 0.99)) * investment

    pa, pb, pc = st.columns(3)
    pa.metric("Daily VaR 95%",          f"-${v95:,.2f}",
              help="On 95% of days, your loss won't exceed this.")
    pb.metric("Daily VaR 99%",          f"-${v99:,.2f}",
              help="On 99% of days, your loss won't exceed this.")
    pc.metric("Expected Annual Profit",  f"+${investment * r_opt:,.0f}",
              help="Based on historical mean returns.")


# ══════════════════════════════════════════════
#  TAB 4 — AI PREDICTOR
# ══════════════════════════════════════════════
with TAB_AI:
    st.subheader("🤖 AI-Powered Price Direction Predictor")
    st.info(
        "**How it works:** A **Random Forest** model (an ensemble of 150 decision trees) "
        "is trained on 12 technical features — RSI, moving-average ratios, momentum, "
        "realised volatility — to predict whether tomorrow's return will be positive or negative."
    )

    selected = st.selectbox("Choose a stock to analyse:", valid_tickers)

    with st.spinner(f"🌲 Training Random Forest on {selected} data  (this takes ~10 s) …"):
        model, scaler, X_te, y_te, y_pred, dir_acc = train_predictor(prices[selected])

    a1, a2, a3 = st.columns(3)
    a1.metric("🎯 Direction Accuracy", f"{dir_acc*100:.1f}%",
              help="% of test days where the model predicted up/down correctly.")
    a2.metric("📊 Test-Set Size",      f"{len(y_te)} days")
    a3.metric("🌲 Trees in Forest",    "150")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔍 Feature Importance")
        feat_names = ["ret","log_ret","ma5_ratio","ma10_ratio","ma20_ratio","ma50_ratio",
                      "vol5","vol20","mom5","mom10","mom20","rsi"]
        imp_df = (pd.DataFrame({"Feature": feat_names,
                                "Importance": model.feature_importances_})
                    .sort_values("Importance"))
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(template="plotly_dark", height=420,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        st.subheader("📈 Predicted vs Actual (last 60 test days)")
        n_plot = min(60, len(y_te))
        dates  = X_te.index[-n_plot:]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=dates, y=y_te.values[-n_plot:] * 100,
            name="Actual Return", line=dict(color="cyan", width=2)
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates, y=y_pred[-n_plot:] * 100,
            name="Predicted Return",
            line=dict(color="orange", width=2, dash="dash")
        ))
        fig_pred.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.25)
        fig_pred.update_layout(template="plotly_dark", height=420,
                               yaxis_title="Daily Return (%)")
        st.plotly_chart(fig_pred, use_container_width=True)

    # ── Next-day prediction ──
    st.subheader("🔮 Next Trading Day Signal")
    feats_all = build_features(prices[selected])
    last_row  = feats_all.drop("target", axis=1).iloc[-1:]
    last_s    = scaler.transform(last_row)
    next_ret  = float(model.predict(last_s)[0])
    last_price = float(prices[selected].iloc[-1])
    pred_price = last_price * (1 + next_ret)

    d1, d2, d3 = st.columns(3)
    direction = "📈 BUY signal" if next_ret > 0 else "📉 SELL signal"
    d1.metric("Signal",          direction)
    d2.metric("Predicted Return", f"{next_ret*100:+.3f}%")
    d3.metric("Predicted Price",  f"${pred_price:.2f}",  delta=f"{next_ret*100:+.3f}%")

    st.warning(
        "⚠️ **Educational use only.** This model is trained purely on price data. "
        "Real markets are driven by macro events, earnings, and sentiment that no price-only "
        "model can fully capture. Never make real investment decisions based solely on ML signals."
    )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "QuantVision · Built with Python, Streamlit, Scikit-learn, Plotly & Markowitz MPT · "
    "Data via Yahoo Finance · Educational purposes only"
    "</div>",
    unsafe_allow_html=True
)