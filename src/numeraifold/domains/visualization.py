"""
Module for visualizing feature domains and feature similarity heatmaps.

This module provides:
- visualize_feature_domains: Creates a scatter plot visualization of feature domains in 2D space.
- visualize_domain_heatmap: Generates a heatmap showing feature similarities grouped by domain.
- create_interactive_domain_visualization: Creates an interactive Plotly-based visualization of feature domains.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def visualize_feature_domains(embedding, cluster_labels, features):
    """
    Visualize feature domains in 2D space.

    Improved with:
    - Error handling for different shapes.
    - Better labeling for readability.
    - Memory optimization for large feature sets.

    Parameters:
        embedding (np.ndarray): 2D array of feature embeddings.
        cluster_labels (np.ndarray): Array of cluster labels for each feature.
        features (list): List of feature names corresponding to embeddings.

    Returns:
        matplotlib.pyplot: The plot object with the visualization, or None if an error occurs.
    """
    if embedding is None or cluster_labels is None:
        print("Error: Empty embedding or cluster labels")
        return None

    try:
        # Ensure the embedding is 2D
        if embedding.shape[1] != 2:
            print(f"Warning: Embedding has {embedding.shape[1]} dimensions, expected 2")
            if embedding.shape[1] > 2:
                embedding = embedding[:, :2]
            else:
                return None

        # Ensure length consistency between embedding, cluster_labels, and features
        if len(embedding) != len(cluster_labels):
            print(f"Error: Embedding length {len(embedding)} ≠ clusters length {len(cluster_labels)}")
            min_len = min(len(embedding), len(cluster_labels))
            embedding = embedding[:min_len]
            cluster_labels = cluster_labels[:min_len]

        if len(embedding) != len(features):
            print(f"Warning: Embedding length {len(embedding)} ≠ features length {len(features)}")
            min_len = min(len(embedding), len(features))
            embedding = embedding[:min_len]
            cluster_labels = cluster_labels[:min_len]
            features = features[:min_len]

        # Limit number of features to display for readability
        max_labels = 100
        show_labels = len(features) <= max_labels
        if not show_labels:
            print(f"Too many features ({len(features)}), limiting labels to {max_labels}")

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=cluster_labels,
            cmap='tab20',
            s=50,
            alpha=0.8
        )

        # Add feature name annotations if the number of features is small enough
        if show_labels:
            for i, feature in enumerate(features):
                feature_label = feature[:20] + '...' if len(feature) > 20 else feature
                plt.annotate(feature_label, (embedding[i, 0], embedding[i, 1]), fontsize=8, alpha=0.7)

        plt.colorbar(scatter, label='Domain ID')
        plt.title('Feature Domains Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()

        return plt

    except Exception as e:
        print(f"Error in visualization: {e}")
        return None

def visualize_domain_heatmap(embedding, cluster_labels, features, figsize=(14, 12), save_path=None):
    """
    Create a heatmap visualization showing feature similarity within and across domains.

    Parameters:
        embedding (np.ndarray): Feature embeddings.
        cluster_labels (np.ndarray): Domain assignments for each feature.
        features (list): Feature names.
        figsize (tuple): Figure size (default: (14, 12)).
        save_path (str): Path to save the visualization (optional).

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure, or None if an error occurs.
    """
    try:
        # Validate inputs: truncate arrays/lists to the shortest length among them
        min_len = min(len(features), len(cluster_labels), len(embedding))
        features = features[:min_len]
        cluster_labels = cluster_labels[:min_len]
        embedding = embedding[:min_len]

        # Calculate pairwise distances and convert to similarities
        distances = squareform(pdist(embedding))
        similarities = 1 / (1 + distances)

        # Sort features based on cluster labels for grouped visualization
        domain_indices = np.argsort(cluster_labels)
        sorted_features = [features[i] for i in domain_indices]
        sorted_domains = cluster_labels[domain_indices]
        sorted_similarities = similarities[domain_indices, :][:, domain_indices]

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            sorted_similarities,
            cmap='viridis',
            xticklabels=sorted_features,
            yticklabels=sorted_features,
            cbar_kws={'label': 'Similarity'}
        )

        # Identify and mark boundaries between different domains
        domain_boundaries = []
        prev_domain = None
        for i, domain in enumerate(sorted_domains):
            if domain != prev_domain:
                domain_boundaries.append(i)
                prev_domain = domain

        # Draw lines to separate domains
        for boundary in domain_boundaries[1:]:
            plt.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=boundary, color='red', linestyle='-', linewidth=1)

        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Feature Similarity Heatmap Grouped by Domain')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")

        return plt.gcf()

    except Exception as e:
        print(f"Error creating domain heatmap: {e}")
        return None

def create_interactive_domain_visualization(embedding, cluster_labels, features,
                                            save_path='feature_domains_interactive.html'):
    """
    Create an interactive visualization of feature domains using Plotly.

    Parameters:
        embedding (np.ndarray): 2D embedding coordinates.
        cluster_labels (np.ndarray): Cluster assignments for each feature.
        features (list): List of feature names.
        save_path (str): Path to save the HTML file (default: 'feature_domains_interactive.html').

    Returns:
        plotly.graph_objects.Figure: The interactive Plotly figure object, or None if an error occurs.
    """
    try:
        # Validate and prepare data by truncating to the shortest length among inputs
        min_len = min(len(features), len(cluster_labels), len(embedding))
        features = features[:min_len]
        cluster_labels = cluster_labels[:min_len]
        embedding = embedding[:min_len]

        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'dimension_1': embedding[:, 0],
            'dimension_2': embedding[:, 1],
            'domain_id': cluster_labels,
            'feature': features
        })

        # Count features per domain and add this info to the DataFrame
        domain_counts = df.groupby('domain_id').size()
        df['domain_size'] = df['domain_id'].map(domain_counts)
        df['hover_text'] = df.apply(
            lambda row: f"Feature: {row['feature']}<br>"
                        f"Domain: {int(row['domain_id'])}<br>"
                        f"Domain size: {int(row['domain_size'])} features",
            axis=1
        )

        # Create interactive scatter plot with Plotly
        fig = px.scatter(
            df,
            x='dimension_1',
            y='dimension_2',
            color='domain_id',
            hover_data=['feature', 'domain_id', 'domain_size'],
            custom_data=['hover_text'],
            title='Interactive Feature Domains Visualization',
            labels={'dimension_1': 'Dimension 1', 'dimension_2': 'Dimension 2'},
            color_continuous_scale=px.colors.qualitative.Bold
        )

        # Update hover information for clarity
        fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

        # Add annotations for each feature
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['dimension_1'],
                y=row['dimension_2'],
                text=row['feature'],
                showarrow=False,
                font=dict(size=8),
                opacity=0.7
            )

        # Update layout for a clean look
        fig.update_layout(
            plot_bgcolor='white',
            hovermode='closest',
            width=1000,
            height=800
        )

        # Save the interactive visualization as an HTML file
        fig.write_html(save_path)
        print(f"Interactive visualization saved to: {save_path}")

        return fig

    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return None
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        return None
