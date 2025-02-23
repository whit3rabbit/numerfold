import numpy as np

from numeraifold.utils.artifacts import load_and_analyze_domains


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
