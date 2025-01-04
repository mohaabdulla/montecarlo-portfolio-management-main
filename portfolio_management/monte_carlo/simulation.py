import numpy as np

class MonteCarloSimulation:
    def __init__(self, returns, initial_investment=1, weights=None):
        self.returns = returns
        # Annualize the mean and covariance properly
        self.mean = returns.mean() * 252  # Annualize mean
        self.covariance = returns.cov() * 252  # Annualize covariance
        self.initial_investment = initial_investment
        num_assets = len(self.mean)
        if weights is None:
            self.weights = np.ones(num_assets) / num_assets
        else:
            self.weights = np.array(weights)

    def run_simulation(self, num_simulations, time_horizon):
        all_cumulative_returns = np.zeros((time_horizon, num_simulations))
        final_portfolio_values = np.zeros(num_simulations)

        # Convert annual parameters to daily
        daily_mean = self.mean / 252
        daily_cov = self.covariance / 252

        for sim in range(num_simulations):
            simulated_returns = np.random.multivariate_normal(
                daily_mean, daily_cov, time_horizon
            )
            # Use log returns for more realistic compounding
            portfolio_returns = simulated_returns.dot(self.weights)
            cumulative_returns = np.exp(np.cumsum(portfolio_returns))
            all_cumulative_returns[:, sim] = cumulative_returns * self.initial_investment
            final_portfolio_values[sim] = cumulative_returns[-1] * self.initial_investment
            
        return all_cumulative_returns, final_portfolio_values