# 📊 Multi-Sector Portfolio Risk & Allocation Analytics Engine

A quantitative portfolio comparison engine built using Python and Streamlit that evaluates sectoral and custom portfolios using Monte Carlo simulation, Modern Portfolio Theory, and advanced risk metrics (Sharpe Ratio, VaR, CVaR).

---

## 🚀 Project Overview

This project enables structured portfolio comparison across predefined Indian sector indices and user-defined portfolios. It is designed as a decision-support analytics tool and does not provide investment recommendations.

---

# ⭐ STAR Framework Explanation

## 🟦 Situation

Retail investors and analysts often compare sector indices and custom portfolios using basic return metrics. However, most tools fail to provide:

- Risk-adjusted comparisons  
- Efficient frontier visualization  
- Monte Carlo-based allocation insights  
- Tail risk metrics (VaR & CVaR)  
- Structured portfolio comparison in one unified system  

There was a need for a professional, quantitative portfolio analytics framework.

---

## 🟨 Task

Design and build a modular portfolio analytics engine that:

- Allows users to analyze custom stock portfolios  
- Supports predefined sector-based portfolios  
- Performs Monte Carlo allocation simulations  
- Computes efficient frontier  
- Identifies Max Sharpe and Min Variance portfolios  
- Calculates VaR and CVaR  
- Enables structured comparison between sectors and custom portfolios  
- Presents results through a clean interactive web interface  

---

## 🟩 Action

Built a fully modular architecture:

### 🔹 Data Layer
- Downloaded historical stock data using `yfinance`
- Computed log returns for stability and time-additivity
- Handled missing data and invalid tickers gracefully

### 🔹 Risk Engine
- Annualized mean return and covariance matrix
- Monte Carlo simulation (25,000 portfolio weight samples)
- Portfolio volatility and Sharpe Ratio computation
- Historical Value at Risk (VaR)
- Conditional Value at Risk (CVaR)

### 🔹 Optimization Layer
- Maximum Sharpe Ratio portfolio (SLSQP optimization)
- Minimum Variance portfolio
- Long-only, fully-invested constraints

### 🔹 Visualization Layer
- Monte Carlo allocation cloud
- Efficient frontier plot
- Monte Carlo simulated path projections
- Portfolio allocation bar charts
- Comparative performance tables

### 🔹 Web Application
- Built using Streamlit
- Wide-layout professional UI
- 4 operating modes:
  1. Custom Portfolio Analysis
  2. Single Sector Analysis
  3. Sector vs Sector Comparison
  4. Custom vs Sector Comparison

---

## 🟪 Result

The final system provides:

- 📈 5 professional visual analytics per portfolio
- 📊 Risk-adjusted performance comparison
- 🧠 Scenario-based portfolio path simulation
- ⚖ Structured risk-return evaluation framework
- 💼 Modular architecture suitable for quantitative finance workflows

The project demonstrates applied knowledge of:

- Portfolio Theory  
- Monte Carlo Simulation  
- Risk Modeling  
- Optimization under constraints  
- Financial Data Engineering  
- Interactive Analytics Deployment  

---

# 📊 Key Features

- Monte Carlo Portfolio Simulation (25,000 simulations)
- Efficient Frontier Construction
- Maximum Sharpe & Minimum Variance Optimization
- VaR (95%) and CVaR (95%) Calculation
- Sector-Level Portfolio Comparison
- Custom Portfolio Evaluation
- Interactive Streamlit Dashboard

---

# 🛠 Tech Stack

- Python
- NumPy
- Pandas
- yfinance
- SciPy
- Matplotlib
- Streamlit

---

# ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
