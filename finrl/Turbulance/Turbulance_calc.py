import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

def calculate_hourly_turbulence_bins(price_df, lookback=5):
    """
    Calculate Mahalanobis distance as turbulence from hourly price data.
    """
    turbulence_scores = []
    timestamps = []

    for i in range(lookback, len(price_df)):
        hist = price_df.iloc[i-lookback:i]
        current = price_df.iloc[i]

        try:
            cov = hist.cov().values
            mean = hist.mean().values
            inv_cov = np.linalg.inv(cov)
            m_dist = mahalanobis(current.values, mean, inv_cov)
        except (np.linalg.LinAlgError, ValueError):
            m_dist = np.nan

        turbulence_scores.append(m_dist)
        timestamps.append(price_df.index[i])

    turbulence_df = pd.DataFrame({
        'turbulence': turbulence_scores
    }, index=pd.to_datetime(timestamps))

    bins = {
        'low': turbulence_df['turbulence'].quantile(0.33),
        'medium': turbulence_df['turbulence'].quantile(0.66)
    }

    return turbulence_df, bins


def assign_turbulence_bins(turbulence_df, bins):
    """
    Label each turbulence score as low/medium/high.
    """
    def label_bin(value):
        if np.isnan(value):
            return 'unknown'
        elif value <= bins['low']:
            return 'low'
        elif value <= bins['medium']:
            return 'medium'
        else:
            return 'high'

    turbulence_df['turbulence_bin'] = turbulence_df['turbulence'].apply(label_bin)
    return turbulence_df


def add_turbulence_to_data(data_df, turbulence_df):
    """
    Merge turbulence scores and bins into main dataset.
    """
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df.set_index('date', inplace=True)
    result = data_df.join(turbulence_df, how='left')
    result.reset_index(inplace=True)
    return result


def analyze_returns_by_turbulence_bin(df):
    """
    Evaluate mean returns per turbulence bin.
    Assumes 'return' and 'turbulence_bin' columns exist.
    """
    return df.groupby('turbulence_bin')['return'].mean().sort_values(ascending=False)


