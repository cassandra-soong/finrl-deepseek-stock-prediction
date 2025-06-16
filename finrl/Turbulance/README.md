1. calculate_hourly_turbulence_bins(price_df, lookback=5)

Purpose
Quantifies how “abnormal” each hour’s price vector is relative to the previous lookback hours using the Mahalanobis distance—a multivariate z-score that incorporates correlations.

Core steps
	1.	Sliding window: For every row from lookback onward, form hist (the trailing window) and current (the point to score).
	2.	Mahalanobis distance:
	•	Compute covariance cov and mean mean of hist.
	•	Invert the covariance (inv_cov).
	•	Distance = mahalanobis(current, mean, inv_cov).
	3.	Error handling: If the covariance is singular or inversion fails, shove in np.nan.
	4.	Build result:
	•	turbulence_df is an indexed Series of distances.
	•	Quantile-cut the series at 33 % and 66 % to produce bins:
        {'low': 33rd-percentile, 'medium': 66th-percentile}

2. assign_turbulence_bins(turbulence_df, bins)

Purpose
Discretises the continuous turbulence score into categorical risk regimes.

Logic
	•	≤ bins['low'] → 'low'
	•	≤ bins['medium'] → 'medium'
	•	bins['medium'] → 'high'
	•	NaN → 'unknown'.

Adds a turbulence_bin column in-place.

⸻

3. add_turbulence_to_data(data_df, turbulence_df)

Purpose
Left-join turbulence information back onto an arbitrary market-level dataframe (data_df) keyed by timestamp.

Steps
	1.	Parse data_df['date'] to datetime, set as index.
	2.	Join (how='left') with turbulence_df so every row inherits its score & bin.
	3.	Restore date to a regular column.

⸻

4. analyze_returns_by_turbulence_bin(df)

Purpose
Diagnostic helper: averages the return column per turbulence regime and ranks them.

Returns a Series sorted descending by mean return.
