import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_model_artifacts(model: torch.nn.Module,
                         save_dir: str,
                         model_config: Dict,
                         feature_list: List[str],
                         target_list: List[str],
                         metrics: Optional[Dict] = None) -> str:
    """
    Save model artifacts including state dict, configuration, and metadata.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        save_dir (str): Directory to save artifacts.
        model_config (Dict): Model configuration dictionary.
        feature_list (List[str]): List of feature names.
        target_list (List[str]): List of target names.
        metrics (Optional[Dict]): Optional dictionary of evaluation metrics.

    Returns:
        str: Path to the saved model directory.
    """
    try:
        # Ensure the base save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique timestamp for this model save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a subdirectory for this model using the timestamp
        model_dir = os.path.join(save_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # Save the model's state dictionary
        model_path = os.path.join(model_dir, "model_state.pt")
        torch.save(model.state_dict(), model_path)

        # Prepare metadata including configuration, features, targets, and metrics
        metadata = {
            'timestamp': timestamp,
            'model_config': model_config,
            'features': feature_list,
            'targets': target_list,
            'metrics': metrics if metrics is not None else {},
            'pytorch_version': torch.__version__
        }

        # Save the metadata as a JSON file
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model artifacts saved to: {model_dir}")
        return model_dir

    except Exception as e:
        print(f"Error saving model artifacts: {str(e)}")
        return ""


def load_model_artifacts():
    """
    Placeholder function to load model artifacts.

    This function should be implemented to load a model's state dictionary,
    configuration, and metadata from a saved directory.

    Returns:
        None
    """
    pass


def save_feature_domains_data(feature_groups, embedding, cluster_labels, features, output_path='feature_domains_data.csv'):
    """
    Save feature domains data to CSV for later analysis.

    Parameters:
        feature_groups (dict): Dictionary mapping domain names to features.
        embedding (np.ndarray): 2D array of embedding coordinates.
        cluster_labels (List[int]): Cluster assignments for each feature.
        features (List[str]): List of feature names.
        output_path (str): Path to save the CSV file.

    Returns:
        str: Path to the saved file.
    """
    try:
        # Initialize list to store feature domain information
        data = []

        # Check if embedding is provided; if not, create basic data with NaNs for embedding dimensions
        if embedding is None or len(embedding) == 0:
            print("Warning: Empty embedding. Creating basic domain data.")
            for domain_name, domain_features in feature_groups.items():
                # Assume domain_name follows a naming convention ending with a numeric id
                domain_id = int(domain_name.split('_')[-1])
                for feature in domain_features:
                    data.append({
                        'feature': feature,
                        'domain_id': domain_id,
                        'domain_name': domain_name,
                        'dimension_1': np.nan,
                        'dimension_2': np.nan
                    })
        else:
            # Ensure consistency by using the shortest length among features, cluster_labels, and embedding
            min_len = min(len(features), len(cluster_labels), len(embedding))
            features = features[:min_len]
            cluster_labels = cluster_labels[:min_len]
            embedding = embedding[:min_len]

            # Verify embedding has at least two dimensions for visualization
            if embedding.shape[1] >= 2:
                for i, feature in enumerate(features):
                    domain_id = cluster_labels[i]
                    domain_name = f"domain_{domain_id}"
                    data.append({
                        'feature': feature,
                        'domain_id': domain_id,
                        'domain_name': domain_name,
                        'dimension_1': embedding[i, 0],
                        'dimension_2': embedding[i, 1] if embedding.shape[1] > 1 else np.nan
                    })
            else:
                print(f"Warning: Embedding has insufficient dimensions: {embedding.shape}")
                # Fallback: assign NaNs for embedding coordinates
                for i, feature in enumerate(features):
                    domain_id = cluster_labels[i]
                    domain_name = f"domain_{domain_id}"
                    data.append({
                        'feature': feature,
                        'domain_id': domain_id,
                        'domain_name': domain_name,
                        'dimension_1': np.nan,
                        'dimension_2': np.nan
                    })

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)

        # Add domain size information by counting features per domain
        domain_sizes = df.groupby('domain_id').size().reset_index(name='domain_size')
        df = df.merge(domain_sizes, on='domain_id', how='left')

        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        print(f"Feature domains data saved to {output_path}")

        # Also save a mapping file for easy domain reference
        domain_mapping = pd.DataFrame([
            {'domain_id': int(domain_name.split('_')[-1]),
             'domain_name': domain_name,
             'feature_count': len(features)}
            for domain_name, features in feature_groups.items()
        ])
        mapping_path = os.path.splitext(output_path)[0] + '_mapping.csv'
        domain_mapping.to_csv(mapping_path, index=False)
        print(f"Domain mapping saved to {mapping_path}")

        return output_path

    except Exception as e:
        print(f"Error saving feature domains data: {e}")
        # Fallback: create a minimal CSV with just the feature groups
        try:
            minimal_data = []
            for domain_name, domain_features in feature_groups.items():
                domain_id = domain_name.split('_')[-1]
                for feature in domain_features:
                    minimal_data.append({
                        'feature': feature,
                        'domain_id': domain_id,
                        'domain_name': domain_name
                    })

            minimal_df = pd.DataFrame(minimal_data)
            fallback_path = 'feature_domains_minimal.csv'
            minimal_df.to_csv(fallback_path, index=False)
            print(f"Minimal feature domains data saved to {fallback_path}")
            return fallback_path

        except Exception as e2:
            print(f"Failed to save even minimal data: {e2}")
            return None


def load_and_analyze_domains(domains_csv_path='feature_domains_data.csv'):
    """
    Load saved domain data from a CSV file and provide analysis utilities.

    Parameters:
        domains_csv_path (str): Path to the CSV file with domain data.

    Returns:
        dict: A dictionary containing analysis results and utility functions.
    """
    try:
        # Load the domain data
        df = pd.read_csv(domains_csv_path)
        print(f"Loaded domain data with {len(df)} features across {df['domain_id'].nunique()} domains")

        # Compute basic statistics for each domain
        domain_stats = df.groupby('domain_id').agg(
            feature_count=('feature', 'count'),
            avg_dim1=('dimension_1', 'mean'),
            avg_dim2=('dimension_2', 'mean')
        ).reset_index()

        print(f"\nDomain statistics:")
        print(domain_stats)

        # Initialize a dictionary to hold analysis results and utilities
        analysis = {}

        # Store the loaded data and computed statistics
        analysis['data'] = df
        analysis['domain_stats'] = domain_stats

        # Utility function: Retrieve features for a specific domain
        def get_domain_features(domain_id):
            return df[df['domain_id'] == domain_id]['feature'].tolist()

        analysis['get_domain_features'] = get_domain_features

        # Utility function: Visualize the domain embeddings with a scatter plot
        def visualize_domains(figsize=(12, 10), save_path=None):
            # Check if embedding dimensions are available
            if 'dimension_1' not in df.columns or df['dimension_1'].isna().all():
                print("Error: No embedding dimensions available for visualization")
                return None

            plt.figure(figsize=figsize)
            scatter = plt.scatter(
                df['dimension_1'],
                df['dimension_2'],
                c=df['domain_id'],
                cmap='tab20',
                s=50,
                alpha=0.8
            )

            # Annotate feature names if the dataset is small
            if len(df) <= 100:
                for i, row in df.iterrows():
                    feature_label = row['feature']
                    if len(feature_label) > 20:
                        feature_label = feature_label[:17] + '...'
                    plt.annotate(
                        feature_label,
                        (row['dimension_1'], row['dimension_2']),
                        fontsize=8,
                        alpha=0.7
                    )

            plt.colorbar(scatter, label='Domain ID')
            plt.title('Feature Domains Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.tight_layout()

            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")

            return plt

        analysis['visualize_domains'] = visualize_domains

        # Utility function: Analyze domain sizes via a bar plot
        def analyze_domain_sizes(figsize=(10, 6), save_path=None):
            plt.figure(figsize=figsize)
            # Sort statistics by feature count for clarity
            sorted_stats = domain_stats.sort_values('feature_count', ascending=False)
            sns.barplot(
                x='domain_id',
                y='feature_count',
                data=sorted_stats
            )
            plt.title('Feature Count by Domain')
            plt.xlabel('Domain ID')
            plt.ylabel('Number of Features')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Domain size analysis saved to {save_path}")

            return plt

        analysis['analyze_domain_sizes'] = analyze_domain_sizes

        # Utility function: Get representative features from each domain
        def get_representative_features(top_n=1):
            """Get representative features from each domain for modeling."""
            representatives = []
            for domain_id in df['domain_id'].unique():
                domain_features = get_domain_features(domain_id)
                # Select up to top_n features from the domain
                representatives.extend(domain_features[:min(top_n, len(domain_features))])
            return representatives

        analysis['get_representative_features'] = get_representative_features

        # Utility function: Create a feature_groups dictionary for the pipeline
        def create_feature_groups():
            """Create feature_groups dictionary from domain data."""
            feature_groups = {}
            for domain_id in df['domain_id'].unique():
                domain_name = f"domain_{domain_id}"
                domain_features = get_domain_features(domain_id)
                feature_groups[domain_name] = domain_features
            return feature_groups

        analysis['create_feature_groups'] = create_feature_groups

        # Attempt an initial visualization of the domains
        try:
            visualize_domains()
        except Exception as e:
            print(f"Initial visualization failed: {e}")

        return analysis

    except Exception as e:
        print(f"Error loading domain data: {e}")
        return None
