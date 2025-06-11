from Turbulance_calc import *
# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from Turbulance_calc import calculate_hourly_turbulence_bins, assign_turbulence_bins, add_turbulence_to_data, analyze_returns_by_turbulence_bin

# Load your hourly price data

nvda_df_yf = pd.read_csv('finrl/trade_data.csv') 

# Assume `nvda_df_yf` is your hourly price data
price_data = nvda_df_yf.pivot(index='date', columns='tic', values='close')
turb_df, bins = calculate_hourly_turbulence_bins(price_data)

# Add bin labels
turb_df = assign_turbulence_bins(turb_df, bins)

# Add turbulence to your main DataFrame
nvda_df_yf = add_turbulence_to_data(nvda_df_yf, turb_df)

# Calculate hourly returns
nvda_df_yf['return'] = nvda_df_yf['close'].pct_change()

# Analyze average return by turbulence bin
bin_performance = analyze_returns_by_turbulence_bin(nvda_df_yf)
print("Performance by Turbulence Bin:\n", bin_performance)

# plot the analysis results
import matplotlib.pyplot as plt
def plot_performance_by_turbulence(bin_performance):
    plt.figure(figsize=(10, 6))
    bin_performance.plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title('Average Return by Turbulence Bin')
    plt.xlabel('Turbulence Bin')
    plt.ylabel('Average Return')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()


import os

# Create directory if it doesn't exist
output_dir = 'finrl/Turbulance'
os.makedirs(output_dir, exist_ok=True)

# Define paths
turbulence_scores_path = os.path.join(output_dir, 'trade_turbulence_scores.csv')
nvda_with_turbulence_path = os.path.join(output_dir, 'trade_nvda_with_turbulence.csv')
performance_path = os.path.join(output_dir, 'trade_bin_performance_analysis.csv')
bins_path = os.path.join(output_dir, 'trade_turbulence_bins.csv')


# Save all files
turb_df.to_csv(turbulence_scores_path)
nvda_df_yf.to_csv(nvda_with_turbulence_path, index=False)
bin_performance.to_csv(performance_path)

# save the plot
plot_performance_by_turbulence(bin_performance)
# Save the bins to a CSV file
bins_df = pd.DataFrame.from_dict(bins, orient='index', columns=['value'])
bins_df.to_csv(bins_path)
# Save the plot as an image file
plt.figure(figsize=(10, 6))
bin_performance.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Average Return by Turbulence Bin')
plt.xlabel('Turbulence Bin')
plt.ylabel('Average Return')
plt.xticks(rotation=0)
plt.grid(axis='y')
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
# Save the plot to the output directory
plt.savefig(os.path.join(output_dir, 'trade_performance_by_turbulence_bin.png'))


# # plot_performance_by_turbulence(bin_performance)
# # Save the turbulence DataFrame to a CSV file
# turb_df.to_csv('finrl-deepseek-stock-prediction/finrl/Turbulance/trade_turbulence_scores.csv')

# # Save the updated main DataFrame with turbulence scores to a CSV file
# nvda_df_yf.to_csv('finrl-deepseek-stock-prediction/finrl/Turbulance/trade_nvda_with_turbulence.csv', index=False)

# # # Save the bins to a CSV file
# # bins_df = pd.DataFrame.from_dict(bins, orient='index', columns=['value'])

# # bins_df.to_csv('finrl/Notebook/turbulence_bins.csv')

# # Save the performance analysis results to a CSV file in the current directory
# # Analyze returns by turbulence bin
# bin_performance.to_csv('finrl-deepseek-stock-prediction/finrl/Turbulance/trade_performance_by_turbulence_bin.csv')

# # save the plot as an image file


# plt.savefig('finrl-deepseek-stock-prediction/finrl/Turbulance/trade_performance_by_turbulence_bin.png')

