import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stock symbol and date range
symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'

# Fetch historical stock price data
data = yf.download(symbol, start=start_date, end=end_date)

# Extract closing prices as a NumPy array
close_prices = data['Close'].to_numpy()

print(close_prices)

def geometric_brownian_motion(mu, sigma, S0, dt, T, num_paths):
    paths = np.zeros((num_paths, int(T/dt)+1))
    paths[:, 0] = S0

    for t in range(1, int(T/dt)+1):
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return paths

# Parameters
log_returns = np.log(close_prices[1:] / close_prices[:-1])
volatility = log_returns.std() * np.sqrt(close_prices.shape[0])
drift = log_returns.mean()
S0 = close_prices[0]  # Initial stock price
dt = 1 / (close_prices.shape[0]-1)  # Time step
T = 1  # Total time
num_paths = 30 # Number of simulated paths

# Simulate paths using GBM
paths = geometric_brownian_motion(drift, volatility, 
                                        S0, dt, T, num_paths)

# Plot the simulated paths
time_steps = np.arange(0, T + dt, dt)
plt.figure(figsize=(10, 6))
GMB_mu = paths.mean(axis=0)
GMB_std = paths.std(axis=0)
plt.fill_between(time_steps, GMB_mu+GMB_std, GMB_mu-GMB_std,
                    color="b", label="GB")
plt.plot(time_steps, close_prices, color="k", label="Real Data")

plt.title('Simulated Geometric Brownian Motion Paths')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
import pdb;pdb.set_trace()
