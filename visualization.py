# ===============================
# VISUALIZATION MODULE
# ===============================

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")


# ====================================
# Efficient Frontier Plot
# ====================================
def plot_efficient_frontier(returns, volatilities,
                            sharpe_ratios,
                            max_sharpe_point,
                            min_var_point):

    plt.figure(figsize=(11, 7))

    # Scatter of Monte Carlo portfolios
    scatter = plt.scatter(volatilities,
                          returns,
                          c=sharpe_ratios,
                          cmap="viridis",
                          alpha=0.5)

    plt.colorbar(scatter, label="Sharpe Ratio")

    # Highlight optimal portfolios
    plt.scatter(max_sharpe_point[1],
                max_sharpe_point[0],
                marker='*',
                s=250,
                label='Max Sharpe')

    plt.scatter(min_var_point[1],
                min_var_point[0],
                marker='o',
                s=200,
                label='Min Variance')

    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.show()


# ====================================
# Monte Carlo Cloud
# ====================================
def plot_monte_carlo_cloud(returns, volatilities, sharpe_ratios):

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(volatilities,
                          returns,
                          c=sharpe_ratios,
                          cmap="plasma",
                          alpha=0.5)

    plt.colorbar(scatter, label="Sharpe Ratio")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Monte Carlo Portfolio Simulation")
    plt.show()


# ====================================
# Return Distribution
# ====================================
def plot_return_distribution(simulated_returns):

    plt.figure(figsize=(10, 6))
    plt.hist(simulated_returns,
             bins=60,
             density=True,
             alpha=0.6)

    plt.title("Distribution of Portfolio Returns")
    plt.show()


# ====================================
# Allocation Bar Chart
# ====================================
def plot_allocation(weights, tickers, title):

    plt.figure(figsize=(10, 5))
    plt.bar(tickers, weights)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


# ====================================
# Monte Carlo Path Plot
# ====================================
def plot_monte_carlo_paths(simulated_paths):

    plt.figure(figsize=(11, 6))

    for path in simulated_paths:
        plt.plot(path, alpha=0.3)

    plt.title("Monte Carlo Simulated Paths")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Growth")
    plt.show()
