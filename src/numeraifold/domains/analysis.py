"""
analysis.py

This module provides functions to analyze feature domain relationships and stability 
across eras, as well as to generate evolutionary profiles for high-performing stocks.

Functions:
    - analyze_domain_relationships: Computes correlations between feature domains and with targets.
    - calculate_feature_stability: Calculates stability metrics for each feature across eras.
    - create_evolutionary_profiles: Generates profiles capturing the evolution of feature distributions
      across performance quantiles.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, skew
from tqdm import tqdm


def analyze_domain_relationships(train_df, features, cluster_labels, targets=None,
                                 output_path='domain_relationships.csv'):
    """
    Analyze relationships between feature domains and optionally with targets.

    This function computes Pearson correlation coefficients between the average values
    of features grouped by domain, and if target columns are provided, correlates domain
    averages with each target. The results are saved to CSV files.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame containing feature data.
        features (list): List of feature names.
        cluster_labels (list): Domain assignments corresponding to each feature.
        targets (list, optional): List of target column names. Defaults to None.
        output_path (str, optional): Path to save the domain relationship CSV file.
                                     Defaults to 'domain_relationships.csv'.

    Returns:
        pd.DataFrame: DataFrame containing domain correlation information.
    """
    try:
        # Create a mapping from each feature to its corresponding domain
        feature_to_domain = {feat: dom for feat, dom in zip(features, cluster_labels)}
        # Obtain unique domains from the cluster labels
        unique_domains = np.unique(cluster_labels)

        # Lists to store correlation results between domain pairs
        domain_pairs = []
        correlations = []
        p_values = []

        # Compute correlations between each pair of domains
        for i, domain1 in enumerate(unique_domains):
            # Get features belonging to the current domain
            domain1_features = [f for f in features if f in feature_to_domain and feature_to_domain[f] == domain1]
            if not domain1_features:
                continue
            # Compute the average (domain center) for features in domain1
            domain1_values = train_df[domain1_features].mean(axis=1)

            # Compare with every subsequent domain to avoid duplicates
            for domain2 in unique_domains[i+1:]:
                domain2_features = [f for f in features if f in feature_to_domain and feature_to_domain[f] == domain2]
                if not domain2_features:
                    continue
                domain2_values = train_df[domain2_features].mean(axis=1)
                try:
                    corr, p_val = pearsonr(domain1_values, domain2_values)
                    domain_pairs.append((domain1, domain2))
                    correlations.append(corr)
                    p_values.append(p_val)
                except Exception:
                    # Skip this pair if the correlation cannot be computed
                    pass

        # Compile the correlation results into a DataFrame
        domain_corr_df = pd.DataFrame({
            'domain1': [p[0] for p in domain_pairs],
            'domain2': [p[1] for p in domain_pairs],
            'correlation': correlations,
            'p_value': p_values,
            'significant': [p < 0.05 for p in p_values]
        })
        # Add a column for the absolute correlation and sort in descending order
        domain_corr_df['abs_corr'] = domain_corr_df['correlation'].abs()
        domain_corr_df = domain_corr_df.sort_values('abs_corr', ascending=False)

        # If target columns are provided, compute domain-target correlations
        if targets:
            target_results = []
            for target in targets:
                if target not in train_df.columns:
                    continue
                target_values = train_df[target].values
                for domain in unique_domains:
                    domain_features = [f for f in features if f in feature_to_domain and feature_to_domain[f] == domain]
                    if not domain_features:
                        continue
                    domain_values = train_df[domain_features].mean(axis=1)
                    try:
                        corr, p_val = pearsonr(domain_values, target_values)
                        target_results.append({
                            'domain': domain,
                            'target': target,
                            'correlation': corr,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        })
                    except Exception:
                        pass
            if target_results:
                target_corr_df = pd.DataFrame(target_results)
                target_corr_df = target_corr_df.sort_values(['target', 'correlation'], ascending=[True, False])
                target_output = output_path.replace('.csv', '_target_correlations.csv')
                target_corr_df.to_csv(target_output, index=False)
                print(f"Domain-target correlations saved to: {target_output}")

        # Save domain-domain correlations
        domain_corr_df.to_csv(output_path, index=False)
        print(f"Domain relationship analysis saved to: {output_path}")

        return domain_corr_df

    except Exception as e:
        print(f"Error analyzing domain relationships: {e}")
        return pd.DataFrame()

def create_evolutionary_profiles(df, features, target_col="target", era_col='era', n_quantiles=10, random_seed=42):
    """
    Generate evolutionary profiles for high-performing stocks based on feature distributions.

    This function segments the data into performance quantiles for each era and computes
    descriptive statistics (mean, standard deviation, count, and skew) for each feature.
    Features are processed in batches to reduce memory usage.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature data, target values, and era information.
        features (list): List of feature names to analyze.
        target_col (str, optional): Column name representing the performance target.
                                    Defaults to MAIN_TARGET.
        era_col (str, optional): Column name representing eras. Defaults to 'era'.
        n_quantiles (int, optional): Number of quantiles to divide the performance metric into.
                                     Defaults to 10.
        random_seed (int, optional): Seed for random operations. Defaults to 42.

    Returns:
        dict: A dictionary mapping each feature to its evolutionary profile DataFrame.
    """
    print("Generating evolutionary profiles...")

    # Ensure the era column exists; if not, create artificial eras
    if era_col not in df.columns:
        print(f"Warning: Era column '{era_col}' not found. Creating artificial eras.")
        df = df.copy()
        df[era_col] = (np.arange(len(df)) // 1000) + 1

    # Validate that the target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return {}

    # Filter to valid features present in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < len(features):
        print(f"Warning: {len(features) - len(valid_features)} features not in DataFrame")
        features = valid_features

    if not features:
        print("No valid features to analyze")
        return {}

    # Get sorted unique eras
    eras = np.sort(df[era_col].unique())

    # Adjust quantiles based on average samples per era
    avg_samples_per_era = len(df) / len(eras)
    if avg_samples_per_era < n_quantiles * 10:
        n_quantiles = max(2, int(avg_samples_per_era / 10))
        print(f"Adjusting quantiles to {n_quantiles} based on data size")

    profiles = {}
    # Process features in small batches to reduce memory consumption
    feature_batches = [features[i:i+5] for i in range(0, len(features), 5)]
    for feature_batch in tqdm(feature_batches, desc="Processing feature batches"):
        for feature in feature_batch:
            try:
                era_profiles = []
                # Process each era for the current feature
                for era in eras:
                    try:
                        era_df = df[df[era_col] == era].copy()
                        if len(era_df) < n_quantiles * 2:
                            # Skip eras with insufficient samples
                            continue
                        # Determine performance quantiles for the target column
                        try:
                            era_df['performance_quantile'] = pd.qcut(
                                era_df[target_col],
                                q=n_quantiles,
                                labels=False,
                                duplicates='drop'
                            )
                        except ValueError:
                            era_df['performance_quantile'] = pd.Series(
                                np.floor(n_quantiles * era_df[target_col].rank(pct=True)),
                                index=era_df.index
                            ).clip(0, n_quantiles-1).astype(int)
                        # Compute descriptive statistics for each quantile group
                        quantile_stats = era_df.groupby('performance_quantile')[feature].agg(
                            ['mean', 'std', 'count']
                        )
                        # Compute skew for each quantile group, if possible
                        try:
                            skew_values = []
                            for q in range(n_quantiles):
                                if q in era_df['performance_quantile'].values:
                                    q_data = era_df[era_df['performance_quantile'] == q][feature].dropna()
                                    if len(q_data) > 2:
                                        skew_values.append(skew(q_data))
                                    else:
                                        skew_values.append(np.nan)
                                else:
                                    skew_values.append(np.nan)
                            quantile_stats['skew'] = skew_values
                        except Exception:
                            pass
                        era_profiles.append(quantile_stats)
                    except Exception as era_error:
                        print(f"Error processing era {era} for feature {feature}: {era_error}")
                        continue
                if era_profiles:
                    try:
                        feature_profile = pd.concat(
                            era_profiles,
                            keys=eras[:len(era_profiles)],
                            names=['era', 'performance_quantile']
                        )
                        profiles[feature] = feature_profile
                    except Exception as concat_error:
                        print(f"Error combining profiles for {feature}: {concat_error}")
            except Exception as feature_error:
                print(f"Error processing feature {feature}: {feature_error}")
                continue

    return profiles

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from feature_engine.selection import SmartCorrelatedSelection

def evaluate_domain_performance(train_df, val_df, domain_features, target_col):
    """
    Train a quick LGBM model on the given domain's features and return the ROC AUC on val_df.
    """
    model = LGBMClassifier(random_state=42)
    model.fit(train_df[domain_features], train_df[target_col])
    preds = model.predict_proba(val_df[domain_features])[:, 1]
    return roc_auc_score(val_df[target_col], preds)

def follow_up_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: dict,
    main_target: str,
    domain_score_threshold: float = 0.51,
    correlation_threshold: float = 0.95
):
    """
    Follow-up pipeline that:
      1) Evaluates each domain's performance with a quick model
      2) Filters out low-performing domains
      3) Prunes highly correlated features within the kept domains
      4) Trains a final global model on the pruned features
      5) Optionally trains a domain-ensemble model for comparison

    Args:
        train_df (pd.DataFrame): Training data (features + target).
        val_df   (pd.DataFrame): Validation data (features + target).
        feature_groups (dict): { 'domain_0': [feat1, feat2, ...], ... }
        main_target (str): Name of the target column in train_df/val_df.
        domain_score_threshold (float): Minimum AUC threshold to keep a domain.
        correlation_threshold (float): Threshold for feature correlation pruning.

    Returns:
        dict: Contains domain_scores, kept_domains, final_model_score,
              ensemble_score (if used), pruned_features, etc.
    """

    # -------------------------------------------------
    # 1. Evaluate domain performance
    # -------------------------------------------------
    print("Evaluating domain performance...")
    domain_scores = {}
    for domain_name, feats in feature_groups.items():
        score = evaluate_domain_performance(train_df, val_df, feats, main_target)
        domain_scores[domain_name] = score
    
    print("Domain performance scores:")
    for d, s in domain_scores.items():
        print(f"  {d}: {s:.4f}")

    # -------------------------------------------------
    # 2. Filter out low-performing domains
    # -------------------------------------------------
    print(f"\nFiltering domains with AUC >= {domain_score_threshold}...")
    kept_domains = [d for d, s in domain_scores.items() if s >= domain_score_threshold]
    print("Kept domains:", kept_domains)

    # Gather all features from kept domains
    kept_features = []
    for d in kept_domains:
        kept_features.extend(feature_groups[d])
    kept_features = list(set(kept_features))  # unique

    # -------------------------------------------------
    # 3. Correlation pruning within kept features
    # -------------------------------------------------
    print(f"\nPruning correlated features (threshold={correlation_threshold})...")
    # Fit the selector on training data's kept features
    selector = SmartCorrelatedSelection(threshold=correlation_threshold, variables=kept_features)
    train_pruned = selector.fit_transform(train_df)
    pruned_features = train_pruned.columns.drop(main_target).tolist()

    # Match columns for validation
    val_pruned = val_df[pruned_features]

    print(f"Reduced from {len(kept_features)} kept features to {len(pruned_features)} after pruning")

    # -------------------------------------------------
    # 4. Train a final global model on pruned features
    # -------------------------------------------------
    print("\nTraining final global model with pruned features...")
    final_model = LGBMClassifier(random_state=42)
    final_model.fit(train_pruned[pruned_features], train_pruned[main_target])
    preds_final = final_model.predict_proba(val_pruned)[:, 1]
    final_model_score = roc_auc_score(val_df[main_target], preds_final)
    print(f"Final global model AUC: {final_model_score:.4f}")

    # -------------------------------------------------
    # 5. (Optional) Train a domain-ensemble model for comparison
    # -------------------------------------------------
    print("\nTraining domain-ensemble models (only for kept domains)...")
    domain_models = {}
    for d in kept_domains:
        feats = list(set(feature_groups[d]).intersection(set(pruned_features)))
        clf = LGBMClassifier(random_state=42)
        clf.fit(train_pruned[feats], train_pruned[main_target])
        domain_models[d] = clf

    # Ensemble predictions (simple average)
    print("Averaging predictions from kept domains...")
    ensemble_preds = np.zeros(len(val_pruned))
    for d in kept_domains:
        feats = list(set(feature_groups[d]).intersection(set(pruned_features)))
        ensemble_preds += domain_models[d].predict_proba(val_pruned[feats])[:, 1]
    ensemble_preds /= len(kept_domains)

    ensemble_score = roc_auc_score(val_df[main_target], ensemble_preds)
    print(f"Domain-ensemble AUC: {ensemble_score:.4f}")

    # -------------------------------------------------
    # 6. Return summary results
    # -------------------------------------------------
    results = {
        'domain_scores': domain_scores,
        'kept_domains': kept_domains,
        'pruned_features': pruned_features,
        'final_model_score': final_model_score,
        'ensemble_score': ensemble_score
    }
    return results
