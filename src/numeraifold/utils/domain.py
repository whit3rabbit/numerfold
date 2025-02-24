import numpy as np
import pandas as pd
import os
from numeraifold.utils.artifacts import load_and_analyze_domains
import os
import pickle
import json

def save_model_and_domains(model, feature_groups, domain_scores, pruned_features, save_dir='./saved_models'):
    """
    Save the trained model and domain data for future use.
    
    Args:
        model: The trained model to save
        feature_groups: Dictionary of feature groups
        domain_scores: Dictionary of domain scores
        pruned_features: List of pruned features
        save_dir: Directory to save the model and data
    
    Returns:
        dict: Dictionary with paths to saved files
    """
   
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save the model using pickle
    if model is not None:
        model_path = os.path.join(save_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        saved_paths['model_path'] = model_path
    
    # Save domain data as JSON
    domain_data = {
        'feature_groups': {k: list(v) for k, v in feature_groups.items()},  # Convert to lists
        'domain_scores': domain_scores,
        'pruned_features': list(pruned_features)  # Convert to list
    }
    
    domain_path = os.path.join(save_dir, 'domains.json')
    with open(domain_path, 'w') as f:
        json.dump(domain_data, f, indent=2)
    saved_paths['domain_path'] = domain_path
    
    print(f"Model and domain data saved to {save_dir}")
    return saved_paths

def load_model_and_domains(load_dir='./saved_models'):
    """
    Load the saved model and domain data.
    
    Args:
        load_dir: Directory where model and data are saved
    
    Returns:
        dict: Dictionary with loaded model and domain data
    """
   
    loaded_data = {}
    
    # Load the model
    model_path = os.path.join(load_dir, 'model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_data['model'] = pickle.load(f)
    
    # Load domain data
    domain_path = os.path.join(load_dir, 'domains.json')
    if os.path.exists(domain_path):
        with open(domain_path, 'r') as f:
            domain_data = json.load(f)
            loaded_data['feature_groups'] = domain_data.get('feature_groups', {})
            loaded_data['domain_scores'] = domain_data.get('domain_scores', {})
            loaded_data['pruned_features'] = domain_data.get('pruned_features', [])
    
    print(f"Model and domain data loaded from {load_dir}")
    return loaded_data

def integrate_domain_data_to_pipeline(run_alphafold_pipeline, domains_csv_path='feature_domains_data.csv'):
    """
    Create a wrapper function that uses saved domain data instead of re-clustering.

    This wrapper leverages cached domain data (loaded from a CSV) to bypass the feature
    re-clustering step in the original AlphaFold pipeline. If cached data is unavailable,
    it falls back to the original pipeline.

    Parameters:
        run_alphafold_pipeline (function): Original AlphaFold pipeline function.
        domains_csv_path (str): Path to the CSV file containing saved domain data.

    Returns:
        function: A wrapped pipeline function that uses cached domain data.
    """
    def wrapped_pipeline(train_df, val_df, features, targets, *args, **kwargs):
        """
        Wrapped pipeline function that leverages cached domain data.

        Attempts to load cached domain data and replaces the global 'identify_feature_domains'
        function with a version that returns the cached data. If any error occurs or if the
        necessary function is not found, the original pipeline is executed.

        Parameters:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            features (list): List of feature names.
            targets (list): List of target names.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of executing the AlphaFold pipeline.
        """
        try:
            # Load cached domain analysis from the provided CSV file.
            domain_analysis = load_and_analyze_domains(domains_csv_path)

            # If loading fails, revert to the original pipeline.
            if domain_analysis is None:
                print("Warning: Could not load domain data. Will re-cluster features.")
                return run_alphafold_pipeline(train_df, val_df, features, targets, *args, **kwargs)

            # Retrieve cached feature groups from the domain analysis.
            feature_groups = domain_analysis['create_feature_groups']()

            # Access the underlying DataFrame containing domain data.
            df = domain_analysis['data']

            # Check if valid embedding dimensions exist to create an embedding and cluster labels.
            if 'dimension_1' in df.columns and not df['dimension_1'].isna().all():
                # Stack dimension_1 and dimension_2 to form a 2D embedding.
                embedding = np.column_stack([df['dimension_1'], df['dimension_2']])
                cluster_labels = df['domain_id'].values
            else:
                embedding = None
                cluster_labels = None

            # Retrieve the original identify_feature_domains function from the pipeline's globals.
            original_identify_domains = run_alphafold_pipeline.__globals__.get('identify_feature_domains')

            def cached_identify_domains(*args, **kwargs):
                """
                Replacement for identify_feature_domains that returns cached domain data.
                """
                print("Using cached domain data instead of re-clustering")
                return feature_groups, embedding, cluster_labels, None

            # If the original function is found, temporarily replace it.
            if original_identify_domains:
                run_alphafold_pipeline.__globals__['identify_feature_domains'] = cached_identify_domains

                try:
                    # Execute the pipeline with the cached domain data.
                    results = run_alphafold_pipeline(train_df, val_df, features, targets, *args, **kwargs)
                finally:
                    # Restore the original identify_feature_domains function.
                    run_alphafold_pipeline.__globals__['identify_feature_domains'] = original_identify_domains

                return results
            else:
                print("Warning: Could not find the identify_feature_domains function. Running original pipeline.")
                return run_alphafold_pipeline(train_df, val_df, features, targets, *args, **kwargs)

        except Exception as e:
            print(f"Error using cached domains: {e}")
            print("Falling back to original pipeline")
            return run_alphafold_pipeline(train_df, val_df, features, targets, *args, **kwargs)

    return wrapped_pipeline

def check_phase1_files(domains_save_path):
    """Check if Phase 1 output files exist."""
    return os.path.exists(domains_save_path)

def load_phase1_data(domains_save_path):
    """Load Phase 1 data from saved domain file."""
    try:
        # Load domain data
        domains_df = pd.read_csv(domains_save_path)
        
        if domains_df.empty:
            print("Warning: Empty domains file")
            return None

        # Initialize variables
        embedding = None
        cluster_labels = None

        # Reconstruct feature groups from the domains file
        feature_groups = {}
        unique_domains = domains_df['domain_id'].unique()
        
        if len(unique_domains) == 0:
            print("Warning: No domains found in file")
            return None
            
        for domain_id in unique_domains:
            domain_name = f"domain_{int(domain_id)}"
            domain_features = domains_df[domains_df['domain_id'] == domain_id]['feature'].tolist()
            if domain_features:  # Only add non-empty domains
                feature_groups[domain_name] = domain_features

        # Load embeddings and cluster labels if available
        if all(col in domains_df.columns for col in ['dimension_1', 'dimension_2']):
            embedding = domains_df[['dimension_1', 'dimension_2']].values
            cluster_labels = domains_df['domain_id'].values

        if not feature_groups:
            print("Warning: No valid feature groups created from domain data")
            return None

        return {
            'feature_groups': feature_groups,
            'embedding': embedding,
            'cluster_labels': cluster_labels,
            'domains_df': domains_df
        }
    except Exception as e:
        print(f"Error loading Phase 1 data: {e}")
        return None
