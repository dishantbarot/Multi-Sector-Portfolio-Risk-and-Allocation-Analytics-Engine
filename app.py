# ==========================================================
# Multi-Sector Portfolio Risk & Allocation Analytics Engine
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(layout="wide")
st.title("📊 Multi-Sector Portfolio Risk & Allocation Analytics Engine")

st.markdown("""
A quantitative portfolio comparison tool using Monte Carlo simulation,
Modern Portfolio Theory, Sharpe Ratio, VaR and CVaR.
""")

st.divider()

# ==========================================================
# PREDEFINED VALID SECTORS
# ==========================================================

SECTORS = {

    "NIFTY BANK": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS",
        "AUBANK.NS", "BANKBARODA.NS", "PNB.NS", "YESBANK.NS"
    ],

    "NIFTY AUTO": [
        "MARUTI.NS", "BAJAJ-AUTO.NS", "M&M.NS",
        "TATAMOTORS.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS"
    ],

    "NIFTY IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS",
        "WIPRO.NS", "TECHM.NS", "LTIM.NS",
        "PERSISTENT.NS"
    ],

    "NIFTY FIN": [
        "JIOFIN.NS", "BAJFINANCE.NS", "HDFCLIFE.NS",
        "SBILIFE.NS", "RECLTD.NS", "PFC.NS",
        "SHRIRAMFIN.NS"
    ],

    "NIFTY METAL": [
        "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
        "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS"
    ],

    "NIFTY ENERGY": [
        "RELIANCE.NS", "NTPC.NS", "ONGC.NS",
        "ADANIPOWER.NS", "BPCL.NS", "POWERGRID.NS"
    ]
}

# ==========================================================
# HELPER FUNCTION: FORMAT USER INPUT
# ==========================================================

def format_tickers_backend(tickers, exchange="NS"):
    formatted = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper()
        if "." not in t:
            t = f"{t}.{exchange}"
        formatted.append(t)
    return formatted


# ==========================================================
# SAFE DATA LOADER (IGNORES INVALID TICKERS)
# ==========================================================

@st.cache_data
def load_data_safe(tickers):

    valid_data = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", progress=False)

            if not df.empty:
                df = df[["Close"]]
                df.columns = [ticker]
                valid_data.append(df)

        except:
            continue

    if not valid_data:
        return None

    data = pd.concat(valid_data, axis=1)
    data = data.dropna(how="all").ffill().dropna()

    return data


# ==========================================================
# CORE ANALYSIS
# ==========================================================

def run_analysis(tickers):

    prices = load_data_safe(tickers)

    if prices is None:
        st.error("No valid tickers found in Yahoo Finance.")
        return None

    log_returns = np.log(prices / prices.shift(1)).dropna()

    mean_returns = log_returns.mean().values * 252
    cov_matrix = log_returns.cov().values * 252

    simulations = 25000
    weights = np.random.dirichlet(np.ones(len(mean_returns)), simulations)

    portfolio_returns = weights @ mean_returns
    portfolio_vol = np.sqrt(
        np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
    )

    sharpe_ratios = portfolio_returns / portfolio_vol

    max_idx = sharpe_ratios.argmax()
    min_idx = portfolio_vol.argmin()

    result = {
        "returns": portfolio_returns,
        "vols": portfolio_vol,
        "sharpes": sharpe_ratios,
        "max_return": portfolio_returns[max_idx],
        "max_vol": portfolio_vol[max_idx],
        "min_return": portfolio_returns[min_idx],
        "min_vol": portfolio_vol[min_idx],
        "max_weights": weights[max_idx],
        "min_weights": weights[min_idx],
        "sharpe": portfolio_returns[max_idx] / portfolio_vol[max_idx],
        "var": np.percentile(portfolio_returns, 5),
        "cvar": portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
        "tickers": prices.columns.tolist()
    }

    return result


# ==========================================================
# MODE SELECTION
# ==========================================================

mode = st.radio(
    "Select Mode",
    [
        "Custom Portfolio",
        "Single Sector",
        "Compare Two Sectors",
        "Compare Custom vs Sector"
    ],
    horizontal=True
)

st.divider()

# ==========================================================
# MODE 1 & 2 — FULL VISUAL ANALYSIS
# ==========================================================

if mode in ["Custom Portfolio", "Single Sector"]:

    if mode == "Custom Portfolio":
        user_input = st.text_input("Enter tickers (comma separated)")
        if st.button("Run Analysis") and user_input:
            tickers = format_tickers_backend(user_input.split(","))

    else:
        sector = st.selectbox("Select Sector", list(SECTORS.keys()))
        if st.button("Run Analysis"):
            tickers = SECTORS[sector]

    if 'tickers' in locals():

        result = run_analysis(tickers)
        if result is None:
            st.stop()

        # ================= METRICS =================
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Return", f"{result['max_return']:.2%}")
        col2.metric("Volatility", f"{result['max_vol']:.2%}")
        col3.metric("Sharpe", f"{result['sharpe']:.2f}")
        col4.metric("VaR (95%)", f"{result['var']:.2%}")
        col5.metric("CVaR (95%)", f"{result['cvar']:.2%}")

        st.divider()

        # ================= VISUAL 1 =================
        fig1, ax1 = plt.subplots()
        sc = ax1.scatter(result["vols"], result["returns"],
                         c=result["sharpes"], cmap="plasma",
                         alpha=0.4, label="Simulated Portfolios")
        ax1.set_title("Monte Carlo Allocation Cloud")
        ax1.set_xlabel("Volatility")
        ax1.set_ylabel("Return")
        ax1.legend()
        fig1.colorbar(sc)
        st.pyplot(fig1)

        # ================= VISUAL 2 =================
        fig2, ax2 = plt.subplots()
        ax2.scatter(result["vols"], result["returns"],
                    alpha=0.3, label="Monte Carlo Portfolios")
        ax2.scatter(result["max_vol"], result["max_return"],
                    marker="*", s=200, color="red", label="Max Sharpe")
        ax2.scatter(result["min_vol"], result["min_return"],
                    marker="o", s=150, color="blue", label="Min Variance")
        ax2.set_title("Efficient Frontier")
        ax2.set_xlabel("Volatility")
        ax2.set_ylabel("Return")
        ax2.legend()
        st.pyplot(fig2)

        # ================= VISUAL 3 =================
        fig3, ax3 = plt.subplots()
        for i in range(50):
            simulated = np.cumprod(
                1 + np.random.normal(
                    result["max_return"]/252,
                    result["max_vol"]/np.sqrt(252),
                    252
                )
            )
            if i == 0:
                ax3.plot(simulated, alpha=0.6, label="Simulated Paths")
            else:
                ax3.plot(simulated, alpha=0.2)
        ax3.set_title("Monte Carlo Path Simulation")
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Portfolio Growth")
        ax3.legend()
        st.pyplot(fig3)

        # ================= VISUAL 4 =================
        fig4, ax4 = plt.subplots()
        ax4.bar(result["tickers"], result["max_weights"],
                label="Max Sharpe Weights")
        ax4.set_title("Max Sharpe Allocation")
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        st.pyplot(fig4)

        # ================= VISUAL 5 =================
        fig5, ax5 = plt.subplots()
        ax5.bar(result["tickers"], result["min_weights"],
                label="Min Variance Weights")
        ax5.set_title("Minimum Variance Allocation")
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend()
        st.pyplot(fig5)


# ==========================================================
# MODE 3 & 4 — COMPARISON TABLE
# ==========================================================

elif mode in ["Compare Two Sectors", "Compare Custom vs Sector"]:

    if mode == "Compare Two Sectors":
        sector1 = st.selectbox("Sector 1", list(SECTORS.keys()))
        sector2 = st.selectbox("Sector 2", list(SECTORS.keys()), index=1)

        if st.button("Compare"):
            r1 = run_analysis(SECTORS[sector1])
            r2 = run_analysis(SECTORS[sector2])

    else:
        sector = st.selectbox("Select Sector", list(SECTORS.keys()))
        user_input = st.text_input("Enter custom tickers")

        if st.button("Compare") and user_input:
            r1 = run_analysis(SECTORS[sector])
            formatted = format_tickers_backend(user_input.split(","))
            r2 = run_analysis(formatted)
            sector1 = sector
            sector2 = "Custom"

    if 'r1' in locals() and r1 and r2:

        comparison = pd.DataFrame({
            "Metric": ["Return", "Volatility", "Sharpe", "VaR", "CVaR"],
            sector1: [
                f"{r1['max_return']:.2%}",
                f"{r1['max_vol']:.2%}",
                f"{r1['sharpe']:.2f}",
                f"{r1['var']:.2%}",
                f"{r1['cvar']:.2%}"
            ],
            sector2: [
                f"{r2['max_return']:.2%}",
                f"{r2['max_vol']:.2%}",
                f"{r2['sharpe']:.2f}",
                f"{r2['var']:.2%}",
                f"{r2['cvar']:.2%}"
            ]
        })

        st.dataframe(comparison, use_container_width=True)
        st.info("Comparison only. No investment recommendation provided.")
