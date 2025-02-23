import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_feature_stability(df, features, era_col='era', window_size=20, random_seed=42):
    """
    Calculate feature stability across eras (evolutionary conservation).

    This function computes various stability metrics for each feature over different eras,
    including:
        - Mean stability: The average rolling standard deviation of era-wise means.
        - Correlation stability: The average rolling correlation of era-wise means.
        - Rank stability: Consistency of feature rankings across eras.
        - Coefficient of variation (cv): Ratio of the standard deviation to the mean of era-wise means.
        - Persistence: A measure of how stable the feature mean is over time.
        - Composite stability score: Mean of the normalized stability metrics.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing feature data.
        features (list): List of feature names to analyze.
        era_col (str): Name of the column representing different eras. Default is 'era'.
        window_size (int): Window size for rolling calculations. Default is 20.
        random_seed (int): Seed for random sampling used in fallback calculations. Default is 42.

    Returns:
        pd.DataFrame: DataFrame with stability metrics for each feature, including a composite stability score.
    """
    print("Calculating feature stability across eras...")

    # Validate that the era column exists; if not, create an arbitrary era assignment
    if era_col not in df.columns:
        print(f"Warning: '{era_col}' not found in DataFrame. Using arbitrary eras.")
        df = df.copy()
        df[era_col] = (np.arange(len(df)) // 1000) + 1

    # Filter features to only those present in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < len(features):
        print(f"Warning: {len(features) - len(valid_features)} features not found in DataFrame")

    # If no valid features are found, return an empty DataFrame with expected columns
    if not valid_features:
        print("Error: No valid features to analyze")
        return pd.DataFrame(columns=[
            'feature', 'mean_stability', 'corr_stability',
            'rank_stability', 'cv', 'persistence', 'stability_score'
        ])

    # Get a sorted array of unique eras from the DataFrame
    eras = np.sort(df[era_col].unique())

    # Adjust window size based on the number of available eras; require at least 2 eras
    window_size = min(window_size, max(2, len(eras) // 2))

    stability_metrics = []

    # Loop through each valid feature with a progress bar
    for feature in tqdm(valid_features, desc="Analyzing features"):
        try:
            # Attempt to compute the era-wise mean for the feature
            try:
                era_means = df.groupby(era_col)[feature].mean()
            except Exception:
                # Fallback: compute era means iteratively
                era_means = pd.Series(
                    [df[df[era_col] == era][feature].mean() for era in eras],
                    index=eras
                )

            # Attempt to compute the era-wise standard deviation for the feature
            try:
                era_stds = df.groupby(era_col)[feature].std()
            except Exception:
                # Fallback: compute era std iteratively
                era_stds = pd.Series(
                    [df[df[era_col] == era][feature].std() for era in eras],
                    index=eras
                )

            # Calculate stability metrics with a rolling window if sufficient data is available
            try:
                if len(era_means) >= window_size:
                    # Mean stability: average rolling standard deviation of era means
                    mean_stability = era_means.rolling(window=window_size).std().mean()
                    # Correlation stability: average rolling correlation of era means
                    corr_stability = era_means.rolling(window=window_size).corr().mean()
                else:
                    # Fallback for insufficient data
                    mean_stability = era_means.std()
                    corr_stability = 0.5  # Neutral value
            except Exception as e:
                print(f"Warning: Error calculating stability for {feature}: {e}")
                mean_stability = np.nan
                corr_stability = np.nan

            # Calculate rank stability using ranking within eras
            try:
                # Rank the feature values as percentiles within each era
                rank_data = df.groupby(era_col)[feature].rank(pct=True)
                # Calculate Spearman correlation of the ranks
                rank_corr = rank_data.corr(method='spearman')
                rank_stability = rank_corr if np.isscalar(rank_corr) else rank_corr.mean()
            except Exception:
                # Fallback: use a sample of the data to compute rank stability
                sample_size = min(10000, len(df))
                sampled_df = df.sample(sample_size, random_state=random_seed)
                try:
                    rank_stability = sampled_df.groupby(era_col)[feature].rank(pct=True).corr(method='spearman')
                except Exception:
                    rank_stability = np.nan

            # Calculate the coefficient of variation (CV) for the era means
            try:
                if np.nanmean(np.abs(era_means)) > 0:
                    cv = np.nanstd(era_means) / np.nanmean(np.abs(era_means))
                else:
                    cv = np.nan
            except Exception:
                cv = np.nan

            # Calculate persistence: a measure of how stable the feature mean is over eras
            try:
                diff_mean = era_means.diff().abs().mean()
                abs_mean = era_means.abs().mean()
                persistence = 1 - (diff_mean / abs_mean if abs_mean > 0 else np.nan)
            except Exception:
                persistence = np.nan

            # Append the computed metrics for the current feature to the list
            stability_metrics.append({
                'feature': feature,
                'mean_stability': mean_stability,
                'corr_stability': corr_stability,
                'rank_stability': rank_stability,
                'cv': cv,
                'persistence': persistence
            })

        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
            # In case of an error, record NaN values for the feature
            stability_metrics.append({
                'feature': feature,
                'mean_stability': np.nan,
                'corr_stability': np.nan,
                'rank_stability': np.nan,
                'cv': np.nan,*************
                'persistence': np.nan
            })

    # Create a DataFrame from the collected stability metrics
    stability_df = pd.DataFrame(stability_metrics)

    # Normalize the computed metrics and calculate a composite stability score
    numeric_cols = ['mean_stability', 'corr_stability', 'rank_stability', 'cv', 'persistence']
    for col in numeric_cols:
        if col in stability_df.columns:
            # Coerce to numeric values and replace infinities with NaN
            stability_df[col] = pd.to_numeric(stability_df[col], errors='coerce')
            # Min-max scale the metric to range between 0 and 1
            col_min = stability_df[col].min()
            col_max = stability_df[col].max()
            if not pd.isna(col_min) and not pd.isna(col_max) and col_max > col_min:
                stability_df[col] = (stability_df[col] - col_min) / (col_max - col_min)

    # Compute the composite stability score as the mean of normalized metrics
    stability_df['stability_score'] = stability_df[numeric_cols].mean(axis=1, skipna=True)
    # Fill any remaining NaN values with a neutral score of 0.5
    stability_df['stability_score'] = stability_df['stability_score'].fillna(0.5)

    return stability_df

def analyze_feature_evolution():
    """
    Placeholder function for analyzing feature evolution.

    This function is intended to extend the analysis of feature stability over time.
    Implementation details should be added as needed.
    """
    pass
