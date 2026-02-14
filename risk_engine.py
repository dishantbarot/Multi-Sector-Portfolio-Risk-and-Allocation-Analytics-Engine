# ===============================
# RISK ENGINE MODULE
# ===============================
# Purpose:
# Perform all mathematical calculations
# used in portfolio analysis.

import numpy as np

TRADING_DAYS = 252   # Standard number of trading days per year


# ====================================
# FUNCTION 1: ANNUALIZED STATISTICS
# ====================================
def annualized_statistics(log_returns):
    """
    Converts daily mean and covariance
    into annual values.
    """

    mean_daily = log_returns.mean().values
    cov_daily = log_returns.cov().values

    # Annualize
    mean_annual = mean_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS

    return mean_annual, cov_annual


# ====================================
# FUNCTION 2: PORTFOLIO PERFORMANCE
# ====================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates:
    - Portfolio expected return
    - Portfolio volatility
    """

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    return portfolio_return, portfolio_vol


# ====================================
# FUNCTION 3: MONTE CARLO (ALLOCATION)
# ====================================
def monte_carlo_simulation(mean_returns, cov_matrix,
                           n_simulations=25000,
                           seed=42):
    """
    Randomly generates many portfolio weight combinations.
    """

    np.random.seed(seed)
    n_assets = len(mean_returns)

    # Random weight generation
    weights = np.random.dirichlet(np.ones(n_assets), n_simulations)

    # Portfolio return for each simulation
    returns = weights @ mean_returns

    # Portfolio volatility
    volatilities = np.sqrt(
        np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
    )

    # Sharpe Ratio (risk-free assumed 0 here)
    sharpe_ratios = returns / volatilities

    return weights, returns, volatilities, sharpe_ratios


# ====================================
# FUNCTION 4: VALUE AT RISK
# ====================================
def value_at_risk(portfolio_returns, alpha=0.95):
    """
    Calculates percentile loss threshold.
    """
    return np.percentile(portfolio_returns, (1 - alpha) * 100)


# ====================================
# FUNCTION 5: EXPECTED SHORTFALL
# ====================================
def expected_shortfall(portfolio_returns, alpha=0.95):
    """
    Calculates average loss beyond VaR.
    """

    var_threshold = value_at_risk(portfolio_returns, alpha)

    return portfolio_returns[
        portfolio_returns <= var_threshold
    ].mean()


# ====================================
# FUNCTION 6: MONTE CARLO PATH SIMULATION
# ====================================
def simulate_portfolio_paths(mean_return,
                             volatility,
                             n_days=252,
                             n_simulations=75,
                             seed=42):
    """
    Simulates future portfolio growth paths
    using Geometric Brownian Motion.
    """

    np.random.seed(seed)
    dt = 1 / 252

    random_shocks = np.random.normal(
        loc=(mean_return - 0.5 * volatility**2) * dt,
        scale=volatility * np.sqrt(dt),
        size=(n_simulations, n_days)
    )

    cumulative_returns = np.exp(np.cumsum(random_shocks, axis=1))

    return cumulative_returns
