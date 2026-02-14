# ===============================
# OPTIMIZATION MODULE
# ===============================
# Purpose:
# Find best portfolio allocations
# using mathematical optimization.

import numpy as np
from scipy.optimize import minimize


# ====================================
# FUNCTION 1: MAX SHARPE PORTFOLIO
# ====================================
def maximize_sharpe(mean_returns, cov_matrix, risk_free_rate=0.02):

    n = len(mean_returns)

    # Objective function (negative Sharpe)
    def neg_sharpe(weights):
        ret = weights @ mean_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(ret - risk_free_rate) / vol

    # Constraint: weights must sum to 1
    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})

    # Bounds: no short selling
    bounds = tuple((0, 1) for _ in range(n))

    initial_guess = np.ones(n) / n

    result = minimize(neg_sharpe,
                      initial_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x


# ====================================
# FUNCTION 2: MINIMUM VARIANCE
# ====================================
def minimize_variance(mean_returns, cov_matrix):

    n = len(mean_returns)

    def portfolio_vol(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0, 1) for _ in range(n))
    initial_guess = np.ones(n) / n

    result = minimize(portfolio_vol,
                      initial_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x
