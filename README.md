# 📊 QuantVision — AI-Powered Portfolio Optimizer

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![License](https://img.shields.io/badge/License-MIT-green)

🌐 **Live Demo:** https://quantvision-ctw9vqxxappnui9ob46tv9v.streamlit.app

> Built by Shivansh Raj | B.Tech Mathematics & Computing | DTU Delhi

## 🏆 What This Does
- 📈 Fetches **real-time stock data** from Yahoo Finance
- 🧮 **Markowitz Mean-Variance Optimization** (Nobel Prize framework) to find optimal portfolio weights
- ⚠️ **Risk Analysis** — VaR, Sharpe Ratio, Beta, Max Drawdown
- 🤖 **Random Forest ML model** trained on 12 technical indicators to predict price direction
- 📊 Interactive dashboard built with Streamlit + Plotly

## 🛠️ Tech Stack
| Layer | Tool |
|-------|------|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Data | yfinance (Yahoo Finance) |
| ML Model | Random Forest (scikit-learn) |
| Optimization | SciPy (SLSQP) |
| Visualisation | Plotly |

## 🚀 Run Locally
```bash
git clone https://github.com/ShivanshRaj18/QuantVision.git
cd QuantVision
pip3 install -r requirements.txt
streamlit run app.py
```

## 📐 Mathematical Concepts
- **Efficient Frontier** — optimal portfolios maximising return per unit risk
- **Sharpe Ratio** = (Return − Risk-Free Rate) / Volatility  
- **Value at Risk (VaR)** — max expected loss at 95%/99% confidence
- **RSI + Momentum** — technical indicators for ML feature engineering
