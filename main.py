# ===============================
# MAIN EXECUTION FILE
# ===============================

# 1. Get user input
# 2. Load data
# 3. Compute returns & risk
# 4. Run Monte Carlo
# 5. Optimize portfolio
# 6. Show plots
# 7. Compare if required

from data_loader import load_data, compute_log_returns, PREDEFINED_INDICES
from risk_engine import *
from optimizer import *
from visualization import *


# ====================================
# Run Portfolio Engine
# ====================================
def run_portfolio_engine(tickers, show_plots=True):

    # Load price data
    prices = load_data(tickers)

    # Convert to log returns
    log_returns = compute_log_returns(prices)

    # Compute annual statistics
    mean_returns, cov_matrix = annualized_statistics(log_returns)

    # Monte Carlo allocation simulation
    weights_mc, returns_mc, vols_mc, sharpe_mc = \
        monte_carlo_simulation(mean_returns, cov_matrix)

    # Optimization
    max_weights = maximize_sharpe(mean_returns, cov_matrix)
    min_weights = minimize_variance(mean_returns, cov_matrix)

    max_perf = portfolio_performance(max_weights,
                                     mean_returns,
                                     cov_matrix)

    # Risk metrics
    var_95 = value_at_risk(returns_mc)
    cvar_95 = expected_shortfall(returns_mc)

    # Plot results
    if show_plots:
        plot_monte_carlo_cloud(returns_mc, vols_mc, sharpe_mc)
        plot_efficient_frontier(returns_mc, vols_mc,
                                sharpe_mc,
                                max_perf,
                                portfolio_performance(min_weights,
                                                      mean_returns,
                                                      cov_matrix))
        plot_allocation(max_weights, tickers,
                        "Max Sharpe Allocation")

    return {
        "return": max_perf[0],
        "volatility": max_perf[1],
        "var_95": var_95,
        "cvar_95": cvar_95
    }
