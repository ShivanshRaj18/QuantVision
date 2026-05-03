# ============================================================
#   QuantVision — AI-Powered Portfolio Optimizer  v2.0
#   Author: Shivansh Raj | DTU Mathematics & Computing
#   NEW in v2: News Sentiment Analysis tab
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
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 QuantVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Portfolio Optimizer · Modern Portfolio Theory + Machine Learning + News Sentiment</div>', unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────
#  SIDEBAR — User Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 📌 Stock Tickers")
    st.caption("🇺🇸 US stocks: AAPL, GOOGL, TSLA")
    st.caption("🇮🇳 Indian stocks: RELIANCE.NS, TCS.NS, INFY.NS")
    tickers_raw = st.text_input(
        "Enter comma-separated tickers",
        value="AAPL, GOOGL, MSFT, AMZN, TSLA",
        help="Add .NS suffix for Indian NSE stocks e.g. TCS.NS"
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
    )

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Data: Yahoo Finance · Model: Random Forest · Math: Markowitz MPT")


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(axis=1, how="all").dropna()


# ─────────────────────────────────────────────
#  PORTFOLIO MATH (Markowitz MPT)
# ─────────────────────────────────────────────
def port_stats(w, mu, cov, rf):
    ret = float(np.dot(w, mu)) * 252
    vol = float(np.sqrt(w @ (cov * 252) @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe


def optimize(returns_df, rf, goal="sharpe"):
    n   = len(returns_df.columns)
    mu  = returns_df.mean()
    cov = returns_df.cov()
    w0  = np.full(n, 1 / n)
    bounds      = [(0.01, 0.60)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    if goal == "sharpe":
        def objective(w):
            return -port_stats(w, mu, cov, rf)[2]
    else:
        def objective(w):
            return port_stats(w, mu, cov, rf)[1]

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    return result.x


def efficient_frontier_cloud(returns_df, rf, n=4000):
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
    return float(np.percentile(ret_series, (1 - conf) * 100))

def max_drawdown(price_series):
    rolling_max = price_series.cummax()
    return float(((price_series - rolling_max) / rolling_max).min())

def beta(stock_ret, bench_ret):
    cov_mat = np.cov(stock_ret, bench_ret)
    return cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 1.0


# ─────────────────────────────────────────────
#  ML PREDICTOR
# ─────────────────────────────────────────────
def build_features(price_s: pd.Series) -> pd.DataFrame:
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

    delta = price_s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df["target"] = ret.shift(-1)
    return df.dropna()


def train_predictor(price_s: pd.Series):
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
#  NEWS SENTIMENT ANALYSIS  ← NEW FEATURE
# ─────────────────────────────────────────────

# Financial keyword lists for sentiment scoring
POSITIVE_WORDS = [
    'surge', 'gain', 'profit', 'growth', 'rise', 'beat', 'strong', 'positive',
    'record', 'buy', 'upgrade', 'outperform', 'rally', 'boost', 'soar', 'high',
    'success', 'exceed', 'win', 'increase', 'advance', 'jump', 'recover',
    'expand', 'bullish', 'breakthrough', 'impressive', 'opportunity', 'peak',
    'revenue', 'dividend', 'acquisition', 'partnership', 'launch', 'approval'
]

NEGATIVE_WORDS = [
    'fall', 'drop', 'loss', 'decline', 'weak', 'miss', 'negative', 'down',
    'cut', 'downgrade', 'underperform', 'sell', 'plunge', 'crash', 'risk',
    'concern', 'fear', 'warn', 'threat', 'decrease', 'slump', 'tumble',
    'bearish', 'struggle', 'problem', 'fail', 'lawsuit', 'investigation',
    'recall', 'layoff', 'bankruptcy', 'debt', 'tariff', 'sanction', 'fine'
]

def analyze_sentiment(text):
    """
    Keyword-based financial sentiment analysis.
    Returns: (label, score, color)
    Score > 0.1  → Bullish
    Score < -0.1 → Bearish
    Otherwise    → Neutral
    """
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    score = (pos - neg) / total if total > 0 else 0.0

    if score > 0.1:
        return "🟢 Bullish", score, "#00C851"
    elif score < -0.1:
        return "🔴 Bearish", score, "#FF4444"
    else:
        return "🟡 Neutral", score, "#FFBB33"


def get_stock_news(ticker, max_items=6):
    """Fetch latest news headlines from Yahoo Finance via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []

        results = []
        for item in raw_news[:max_items]:
            # yfinance changed its news format — handle both old & new
            try:
                content = item.get('content', {})
                if isinstance(content, dict) and content:
                    title     = content.get('title', '')
                    publisher = content.get('provider', {}).get('displayName', '') \
                                if isinstance(content.get('provider'), dict) else ''
                    link      = content.get('canonicalUrl', {}).get('url', '') \
                                if isinstance(content.get('canonicalUrl'), dict) else ''
                else:
                    title     = item.get('title', '')
                    publisher = item.get('publisher', '')
                    link      = item.get('link', '')
            except Exception:
                title     = item.get('title', '')
                publisher = item.get('publisher', '')
                link      = item.get('link', '')

            if not title:
                continue

            label, score, color = analyze_sentiment(title)
            results.append({
                'title':     title,
                'label':     label,
                'score':     score,
                'color':     color,
                'link':      link,
                'publisher': publisher
            })

        return results
    except Exception:
        return []


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
with st.spinner("📡 Fetching market data from Yahoo Finance …"):
    try:
        prices = load_prices(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

valid_tickers = list(prices.columns)
if not valid_tickers:
    st.error("No valid tickers found. Please check your input.")
    st.stop()

daily_ret = prices.pct_change().dropna()
COLORS    = px.colors.qualitative.Plotly

# ── 5 tabs now (added News Sentiment) ──
TAB_OVERVIEW, TAB_OPTIMIZER, TAB_RISK, TAB_AI, TAB_NEWS = st.tabs([
    "📈  Market Overview",
    "🎯  Portfolio Optimizer",
    "⚠️  Risk Analysis",
    "🤖  AI Predictor",
    "📰  News Sentiment",        # ← NEW
])


# ══════════════════════════════════════════════
#  TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════
with TAB_OVERVIEW:
    st.subheader("📈 Live Stock Performance")

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

    st.subheader("📋 Summary Statistics")
    stats = pd.DataFrame({
        "Ticker":          valid_tickers,
        "Total Return":    [f"{(prices[t].iloc[-1]/prices[t].iloc[0]-1)*100:.2f}%" for t in valid_tickers],
        "Ann. Return":     [f"{daily_ret[t].mean()*252*100:.2f}%"                  for t in valid_tickers],
        "Ann. Volatility": [f"{daily_ret[t].std()*np.sqrt(252)*100:.2f}%"          for t in valid_tickers],
        "Best Day":        [f"{daily_ret[t].max()*100:.2f}%"                       for t in valid_tickers],
        "Worst Day":       [f"{daily_ret[t].min()*100:.2f}%"                       for t in valid_tickers],
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📈 Expected Annual Return",   f"{r_opt*100:.2f}%")
    m2.metric("📉 Annual Volatility (Risk)",  f"{v_opt*100:.2f}%")
    m3.metric("⭐ Sharpe Ratio",               f"{s_opt:.3f}")
    m4.metric("💰 Expected 1-yr Gain",        f"${investment*r_opt:,.0f}")

    st.divider()
    c1, c2 = st.columns([1.3, 0.7])

    with c1:
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
        fig_pie.update_layout(template="plotly_dark", height=430, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("💼 Detailed Allocation")
    alloc = pd.DataFrame({
        "Stock":                  valid_tickers,
        "Weight":                 [f"{w*100:.1f}%"          for w in w_opt],
        "Amount Invested":        [f"${w*investment:,.2f}"  for w in w_opt],
        "Annual Return Contrib.": [f"{w * daily_ret[t].mean() * 252 * 100:.2f}%"
                                   for w, t in zip(w_opt, valid_tickers)],
    })
    st.dataframe(alloc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 3 — RISK ANALYSIS
# ══════════════════════════════════════════════
with TAB_RISK:
    st.subheader("⚠️ Comprehensive Risk Dashboard")

    with st.spinner("📡 Loading S&P 500 benchmark (SPY)…"):
        try:
            bench_prices = load_prices(["SPY"], start_date, end_date)
            bench_ret    = bench_prices.pct_change().dropna()["SPY"]
            has_bench    = True
        except Exception:
            has_bench = False

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

    st.subheader(f"💼 Portfolio Value-at-Risk  (Investment = ${investment:,.0f})")
    port_r = daily_ret[valid_tickers].dot(w_opt)
    v95 = abs(value_at_risk(port_r, 0.95)) * investment
    v99 = abs(value_at_risk(port_r, 0.99)) * investment

    pa, pb, pc = st.columns(3)
    pa.metric("Daily VaR 95%",         f"-${v95:,.2f}")
    pb.metric("Daily VaR 99%",         f"-${v99:,.2f}")
    pc.metric("Expected Annual Profit", f"+${investment * r_opt:,.0f}")


# ══════════════════════════════════════════════
#  TAB 4 — AI PREDICTOR
# ══════════════════════════════════════════════
with TAB_AI:
    st.subheader("🤖 AI-Powered Price Direction Predictor")
    st.info(
        "**How it works:** A **Random Forest** model (150 decision trees) "
        "trained on 12 technical features — RSI, moving-average ratios, momentum, "
        "realised volatility — predicts whether tomorrow's return will be positive or negative."
    )

    selected = st.selectbox("Choose a stock to analyse:", valid_tickers)

    with st.spinner(f"🌲 Training Random Forest on {selected} data …"):
        model, scaler, X_te, y_te, y_pred, dir_acc = train_predictor(prices[selected])

    a1, a2, a3 = st.columns(3)
    a1.metric("🎯 Direction Accuracy", f"{dir_acc*100:.1f}%")
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

    st.subheader("🔮 Next Trading Day Signal")
    feats_all = build_features(prices[selected])
    last_row  = feats_all.drop("target", axis=1).iloc[-1:]
    last_s    = scaler.transform(last_row)
    next_ret  = float(model.predict(last_s)[0])
    last_price = float(prices[selected].iloc[-1])
    pred_price = last_price * (1 + next_ret)

    d1, d2, d3 = st.columns(3)
    direction = "📈 BUY signal" if next_ret > 0 else "📉 SELL signal"
    d1.metric("Signal",           direction)
    d2.metric("Predicted Return",  f"{next_ret*100:+.3f}%")
    d3.metric("Predicted Price",   f"${pred_price:.2f}", delta=f"{next_ret*100:+.3f}%")

    st.warning("⚠️ Educational use only. Never make real investment decisions based solely on ML signals.")


# ══════════════════════════════════════════════
#  TAB 5 — NEWS SENTIMENT  ← NEW FEATURE
# ══════════════════════════════════════════════
with TAB_NEWS:
    st.subheader("📰 AI News Sentiment Analyzer")
    st.info(
        "**How it works:** Fetches the latest news headlines for each stock from Yahoo Finance, "
        "then uses **financial keyword-based sentiment analysis** to classify each headline "
        "as 🟢 Bullish, 🔴 Bearish, or 🟡 Neutral. "
        "An aggregate sentiment score is calculated for each stock."
    )

    # ── Fetch all news upfront ──
    all_news = {}
    with st.spinner("📡 Fetching latest news for all stocks…"):
        for tkr in valid_tickers:
            all_news[tkr] = get_stock_news(tkr, max_items=7)

    # ── Overall sentiment summary row ──
    st.subheader("📊 Overall Sentiment Summary")
    sent_cols = st.columns(len(valid_tickers))

    for i, tkr in enumerate(valid_tickers):
        with sent_cols[i]:
            news_list = all_news[tkr]
            if news_list:
                avg_score = float(np.mean([n["score"] for n in news_list]))
                if avg_score > 0.1:
                    overall_label = "🟢 Bullish"
                    delta_color   = "normal"
                elif avg_score < -0.1:
                    overall_label = "🔴 Bearish"
                    delta_color   = "inverse"
                else:
                    overall_label = "🟡 Neutral"
                    delta_color   = "off"
                st.metric(
                    label=f"**{tkr}**",
                    value=overall_label,
                    delta=f"Avg score: {avg_score:+.2f}",
                    delta_color=delta_color
                )
            else:
                st.metric(label=f"**{tkr}**", value="No news")

    st.divider()

    # ── Sentiment bar chart ──
    st.subheader("📈 Sentiment Score Comparison")
    chart_rows = []
    for tkr in valid_tickers:
        nl = all_news[tkr]
        if nl:
            chart_rows.append({"Stock": tkr,
                                "Score": float(np.mean([n["score"] for n in nl]))})

    if chart_rows:
        cdf = pd.DataFrame(chart_rows)
        bar_colors = ["#00C851" if s > 0.1 else "#FF4444" if s < -0.1 else "#FFBB33"
                      for s in cdf["Score"]]
        fig_sent = go.Figure(go.Bar(
            x=cdf["Stock"], y=cdf["Score"],
            marker_color=bar_colors,
            text=[f"{s:+.3f}" for s in cdf["Score"]],
            textposition="outside"
        ))
        fig_sent.add_hline(y=0.1,  line_dash="dot", line_color="#00C851", opacity=0.4,
                           annotation_text="Bullish threshold")
        fig_sent.add_hline(y=-0.1, line_dash="dot", line_color="#FF4444", opacity=0.4,
                           annotation_text="Bearish threshold")
        fig_sent.update_layout(
            template="plotly_dark", height=320,
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1.2, 1.2]),
            showlegend=False
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()

    # ── Detailed news for selected stock ──
    selected_news = st.selectbox(
        "📌 Select a stock to read its latest news:",
        valid_tickers, key="news_sel"
    )

    st.subheader(f"📰 Latest Headlines — {selected_news}")
    headlines = all_news.get(selected_news, [])

    if headlines:
        for item in headlines:
            col_icon, col_text = st.columns([0.12, 0.88])
            with col_icon:
                st.markdown(f"## {item['label'].split()[0]}")   # emoji only
            with col_text:
                if item["link"]:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                else:
                    st.markdown(f"**{item['title']}**")
                pub = item.get("publisher", "")
                score_pct = item["score"]
                detail = f"Sentiment score: **{score_pct:+.2f}**"
                if pub:
                    detail += f"  ·  Source: *{pub}*"
                st.caption(detail)
            st.divider()
    else:
        st.info(f"No recent news found for **{selected_news}**. "
                "This can happen for less-traded or international stocks. "
                "Try switching to a different stock.")

    st.caption(
        "ℹ️ Sentiment is computed using financial keyword matching on Yahoo Finance headlines. "
        "Scores range from –1 (very bearish) to +1 (very bullish). For research purposes only."
    )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "QuantVision v2.0 · Python · Streamlit · Scikit-learn · Plotly · Markowitz MPT · "
    "News Sentiment Analysis · Data via Yahoo Finance · Educational purposes only"
    "</div>",
    unsafe_allow_html=True
)
