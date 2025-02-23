"""
execution.py

This module implements the main pipeline execution functions for NumerAIFold,
including the full pipeline (feature domain identification, model training,
feature generation, and final evaluation) and a domains-only pipeline.
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import traceback

# Configuration and seed settings
from numeraifold.config import RANDOM_SEED
from numeraifold.utils.seed import set_seed

# Core modules for model, training, and evaluation
from numeraifold.core.model import NumerAIFold
from numeraifold.core.training import train_numeraifold_model
from numeraifold.core.evaluation import run_final_evaluation, print_evaluation_results

# Domain identification, visualization, and analysis
from numeraifold.domains.identification import identify_feature_domains, create_sequence_representation
from numeraifold.domains.visualization import visualize_feature_domains, visualize_domain_heatmap, create_interactive_domain_visualization
from numeraifold.domains.analysis import create_evolutionary_profiles, analyze_domain_relationships

# Feature engineering and stability analysis
from numeraifold.features.engineering import generate_alphafold_features
from numeraifold.features.stability import calculate_feature_stability

# Utilities for saving/loading artifacts
from numeraifold.utils.artifacts import save_model_artifacts, save_feature_domains_data, load_and_analyze_domains

# Download numerai data
from numeraifold.data.loading import load_data

# Load the numerai data for transformer
from numeraifold.data.dataloader import get_dataloaders


def run_alphafold_pipeline(train_df, val_df, features, targets,
                           n_clusters=10, confidence_threshold=0.5,
                           batch_size=64, epochs=10, embed_dim=256,
                           num_layers=4, num_heads=8, random_seed=RANDOM_SEED,
                           save_domains=True, domains_save_path='feature_domains_data.csv',
                           force_phase1=False, base_path='.', save_model=True,
                           skip_phase1=False):
    """
    Run the complete AlphaFold-inspired pipeline for Numerai.

    Parameters:
        train_df (DataFrame): Training data.
        val_df (DataFrame): Validation data.
        features (list): List of feature names.
        targets (list): List of target column names.
        n_clusters (int): Number of clusters for feature domain identification.
        confidence_threshold (float): Threshold for feature generation confidence.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        embed_dim (int): Embedding dimensions.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        random_seed (int): Random seed for reproducibility.
        save_domains (bool): Whether to save domain data.
        domains_save_path (str): File path to save domain data.
        force_phase1 (bool): If True, force running Phase 1.
        base_path (str): Base path for saving artifacts.
        save_model (bool): Whether to save the trained model.
        skip_phase1 (bool): If True, skip Phase 1 and load cached domain data.

    Returns:
        dict: Results dictionary containing domain data, trained model, evaluation metrics, etc.
    """
    print("Starting AlphaFold-inspired Numerai pipeline...")

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    results = {}
    embedding = None
    cluster_labels = None
    feature_groups = None

    try:
        # ----- Phase 1: Feature Domain Identification -----
        print("----- Phase 1: Feature Domain Identification -----")
        if skip_phase1:
            print(f"Skipping Phase 1. Loading domain data from: {domains_save_path}")
            # Use the utility to load domain data
            phase1_data = load_and_analyze_domains(domains_save_path)
            if phase1_data is not None:
                results.update(phase1_data)
                feature_groups = phase1_data.get('feature_groups')
                embedding = phase1_data.get('embedding')
                cluster_labels = phase1_data.get('cluster_labels')
                print(f"Successfully loaded {len(feature_groups)} feature groups from saved data.")
            else:
                print(f"Error loading domain data from {domains_save_path}. Proceeding with Phase 1.")
                skip_phase1 = False  # Fallback to running Phase 1 if loading fails

        if not skip_phase1:
            # Check if cached domain data exists using os.path.exists
            if not force_phase1 and os.path.exists(domains_save_path):
                print(f"Found existing domain data at {domains_save_path}. Loading data...")
                phase1_data = load_and_analyze_domains(domains_save_path)
                if phase1_data is not None:
                    results.update(phase1_data)
                    feature_groups = phase1_data.get('feature_groups')
                    embedding = phase1_data.get('embedding')
                    cluster_labels = phase1_data.get('cluster_labels')
                    print(f"Successfully loaded {len(feature_groups)} feature groups")
                else:
                    print("Failed to load Phase 1 data. Running Phase 1...")
                    force_phase1 = True
            else:
                if force_phase1:
                    print("Force running Phase 1...")
                else:
                    print("No existing domain data found. Running Phase 1...")
                force_phase1 = True

            if force_phase1:
                # Identify feature domains using dimensionality reduction and clustering
                feature_groups, embedding, cluster_labels, _ = identify_feature_domains(
                    train_df, features, n_clusters=n_clusters, random_state=random_seed
                )
                results['feature_groups'] = feature_groups

                # Save domain data if requested
                if save_domains:
                    try:
                        saved_path = save_feature_domains_data(
                            feature_groups, embedding, cluster_labels, features,
                            output_path=domains_save_path
                        )
                        if saved_path:
                            results['domains_saved_path'] = saved_path
                            print(f"Feature domain data saved to: {saved_path}")
                    except Exception as e:
                        print(f"Warning: Failed to save domain data: {e}")

            # Ensure we have feature groups even if identification failed
            if feature_groups is None:
                print("Warning: No feature groups created. Using single group.")
                feature_groups = {"domain_0": features}
                results['feature_groups'] = feature_groups

            # Create domain visualization if embedding and cluster labels exist
            if embedding is not None and cluster_labels is not None:
                try:
                    domain_plot = visualize_feature_domains(embedding, cluster_labels, features)
                    results['domain_plot'] = domain_plot
                except Exception as e:
                    print(f"Warning: Domain visualization failed: {e}")

            # Calculate feature stability
            try:
                stability_df = calculate_feature_stability(train_df, features)
                results['stability_df'] = stability_df
            except Exception as e:
                print(f"Warning: Feature stability calculation failed: {e}")
                stability_df = pd.DataFrame({'feature': features})
                results['stability_df'] = stability_df

            # Create sequence representation with memory optimization
            try:
                print("Creating sequence representation (this may take a while)...")
                if len(train_df) > 20000:
                    print(f"Using {min(20000, len(train_df))} samples for sequence representation")
                    seq_df = train_df.sample(min(20000, len(train_df)), random_state=random_seed)
                else:
                    seq_df = train_df

                sequences = create_sequence_representation(seq_df, feature_groups)
                # Store a subset of sequences for inspection
                results['sample_sequences'] = {k: sequences[k] for k in list(sequences.keys())[:5]} if sequences else {}
            except Exception as e:
                print(f"Warning: Sequence representation failed: {e}")
                sequences = {}

            # Generate evolutionary profiles (skipped for large datasets)
            try:
                if len(train_df) > 50000:
                    print("Skipping evolutionary profiles generation for large dataset")
                    profiles = {}
                else:
                    profiles = create_evolutionary_profiles(train_df, features)
                results['sample_profiles'] = {k: profiles[k] for k in list(profiles.keys())[:3]} if profiles else {}
            except Exception as e:
                print(f"Warning: Evolutionary profiles generation failed: {e}")
                profiles = {}
        else:
            # If Phase 1 is completely skipped
            print("Phase 1 fully skipped.")

        # ----- Phase 2: Model Architecture -----
        print("----- Phase 2: Model Architecture -----")

        # Validate features exist in both train and validation sets
        valid_features = [f for f in features if f in train_df.columns and f in val_df.columns]
        if len(valid_features) < len(features):
            print(f"Warning: {len(features) - len(valid_features)} features not found in both train and val datasets")
            features = valid_features

        if not features:
            raise ValueError("No valid features for model training")

        # Instantiate the NumerAIFold model with the specified architecture
        model = NumerAIFold(
            num_features=len(features),
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        results['model_config'] = {
            'num_features': len(features),
            'num_layers': num_layers,
            'embed_dim': embed_dim,
            'num_heads': num_heads
        }

        # Prepare data loaders with error handling
        try:
            # If a custom dataloader function exists in the global scope, use it
            if 'get_dataloaders' in globals() and callable(globals()['get_dataloaders']):
                train_loader, val_loader = get_dataloaders(
                    train_df, val_df, features, targets, batch_size=batch_size
                )
            else:
                # Fallback to manual dataloader creation
                X_train = np.array(train_df[features].fillna(0).values, dtype=np.float32)
                X_val = np.array(val_df[features].fillna(0).values, dtype=np.float32)

                y_train = np.array(train_df[targets[0]].fillna(0).values, dtype=np.float32).reshape(-1, 1)
                y_val = np.array(val_df[targets[0]].fillna(0).values, dtype=np.float32).reshape(-1, 1)

                # Convert to tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

                # Create TensorDatasets and DataLoaders
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        except Exception as e:
            print(f"Error in dataloader creation: {e}")
            print("Attempting to create simpler dataloaders...")
            try:
                X_train_np = np.array(train_df[features].fillna(0).values, dtype=np.float32)
                y_train_np = np.array(train_df[targets[0]].fillna(0).values, dtype=np.float32).reshape(-1, 1)
                X_val_np = np.array(val_df[features].fillna(0).values, dtype=np.float32)
                y_val_np = np.array(val_df[targets[0]].fillna(0).values, dtype=np.float32).reshape(-1, 1)

                X_train = torch.tensor(X_train_np, dtype=torch.float32)
                y_train = torch.tensor(y_train_np, dtype=torch.float32)
                X_val = torch.tensor(X_val_np, dtype=torch.float32)
                y_val = torch.tensor(y_val_np, dtype=torch.float32)

                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            except Exception as e2:
                print(f"Fallback dataloader creation also failed: {e2}")
                print("Creating minimal emergency dataloaders")
                feature = features[0]
                X_train_min = torch.tensor(train_df[feature].fillna(0).values, dtype=torch.float32).view(-1, 1)
                y_train_min = torch.tensor(train_df[targets[0]].fillna(0).values, dtype=torch.float32).view(-1, 1)
                X_val_min = torch.tensor(val_df[feature].fillna(0).values, dtype=torch.float32).view(-1, 1)
                y_val_min = torch.tensor(val_df[targets[0]].fillna(0).values, dtype=torch.float32).view(-1, 1)

                train_dataset = torch.utils.data.TensorDataset(X_train_min, y_train_min)
                val_dataset = torch.utils.data.TensorDataset(X_val_min, y_val_min)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ----- Model Training -----
        try:
            print(f"Training model on {len(features)} features for {epochs} epochs...")
            trained_model = train_numeraifold_model(model, train_loader, val_loader, epochs=epochs)
            results['trained_model'] = trained_model
        except Exception as e:
            print(f"Error in model training: {e}")
            print("Attempting to train a simpler model...")
            simple_model = NumerAIFold(
                num_features=len(features),
                num_layers=2,    # Fewer layers
                embed_dim=64,    # Smaller embeddings
                num_heads=4      # Fewer attention heads
            )
            try:
                trained_model = train_numeraifold_model(simple_model, train_loader, val_loader, epochs=max(3, epochs // 2))
                results['trained_model'] = trained_model
            except Exception as e2:
                print(f"Error training simpler model: {e2}")
                print("Skipping model training step")
                results['trained_model'] = None
                # Create a baseline model for subsequent steps
                baseline_model = NumerAIFold(
                    num_features=len(features),
                    num_layers=1,
                    embed_dim=32,
                    num_heads=2
                )
                baseline_model.eval()
                trained_model = baseline_model

        # If only domain data is needed and no model is trained, exit early
        if results.get('domains_saved_path') and not trained_model:
            print("Domain data saved. Skipping remaining steps as requested.")
            return results

        # ----- Phase 3: Feature Generation -----
        print("----- Phase 3: Feature Generation -----")
        if trained_model is not None:
            try:
                print("Generating AlphaFold-inspired features...")
                train_features_df, val_features_df = generate_alphafold_features(
                    train_df, val_df, trained_model, features,
                    confidence_threshold=confidence_threshold
                )
                if train_features_df is not None and val_features_df is not None:
                    print(f"Train features shape: {train_features_df.shape}")
                    print(f"Val features shape: {val_features_df.shape}")
                    train_enhanced = pd.concat([train_df, train_features_df], axis=1)
                    val_enhanced = pd.concat([val_df, val_features_df], axis=1)
                    results['train_enhanced'] = train_enhanced
                    results['val_enhanced'] = val_enhanced
                else:
                    raise Exception("Feature generation returned None")
            except Exception as e:
                print(f"Error in feature generation: {e}")
                print("Using original features without enhancement")
                train_enhanced = train_df.copy()
                val_enhanced = val_df.copy()
                # Add dummy columns to maintain pipeline compatibility
                train_enhanced['af_confidence'] = 0.5
                train_enhanced['prediction'] = train_enhanced[targets[0]].mean()
                train_enhanced['af_high_confidence'] = 0
                val_enhanced['af_confidence'] = 0.5
                val_enhanced['prediction'] = val_enhanced[targets[0]].mean()
                val_enhanced['af_high_confidence'] = 0
        else:
            print("Skipping feature generation due to model training failure")
            train_enhanced = train_df.copy()
            val_enhanced = val_df.copy()
            train_enhanced['af_confidence'] = 0.5
            train_enhanced['prediction'] = train_enhanced[targets[0]].mean()
            train_enhanced['af_high_confidence'] = 0
            val_enhanced['af_confidence'] = 0.5
            val_enhanced['prediction'] = val_enhanced[targets[0]].mean()
            val_enhanced['af_high_confidence'] = 0

        # ----- Phase 4: Final Evaluation -----
        print("----- Phase 4: Final Evaluation -----")
        try:
            # Identify generated feature columns (if any)
            generated_cols = [c for c in train_enhanced.columns if c.startswith('af_emb_') and c in val_enhanced.columns]
            print(f"Found {len(generated_cols)} generated feature columns")

            # Check for required columns
            required_cols = ['prediction', 'af_confidence', 'af_high_confidence']
            missing_cols = [col for col in required_cols if col not in train_enhanced.columns or col not in val_enhanced.columns]
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")

            # Build the final feature set
            if generated_cols:
                print(f"Using {len(generated_cols)} generated features for final model")
                all_features = features + generated_cols
            else:
                print("No generated features available, using original features")
                all_features = features

            # Ensure features exist in both datasets
            all_features = [f for f in all_features if f in train_enhanced.columns and f in val_enhanced.columns]
            print(f"Final feature set contains {len(all_features)} features")
            print("Running final model evaluation...")

            # Run evaluation on the final model
            eval_results = run_final_evaluation(
                val_df=val_enhanced,
                model=trained_model,
                val_loader=val_loader,
                targets=targets,
                era_col='era',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            if eval_results is not None:
                results['evaluation'] = eval_results
                if 'val_df' in eval_results:
                    val_enhanced = eval_results['val_df']
                    results['val_enhanced'] = val_enhanced
                if 'standard_metrics' in eval_results:
                    results['results_standard'] = eval_results['standard_metrics']
                if 'weighted_metrics' in eval_results:
                    results['results_weighted'] = eval_results['weighted_metrics']
                results['final_features'] = all_features

                print_evaluation_results(eval_results)

                # Save model artifacts if requested
                if save_model:
                    try:
                        model_dir = save_model_artifacts(
                            model=trained_model,
                            save_dir=os.path.join(base_path, 'models'),
                            model_config=results['model_config'],
                            feature_list=features,
                            target_list=targets if isinstance(targets, list) else [targets],
                            metrics=eval_results.get('standard_metrics')
                        )
                        results['model_dir'] = model_dir
                    except Exception as save_error:
                        print(f"Error saving model artifacts: {save_error}")
        except Exception as e:
            print(f"Error in final model evaluation: {e}")
            print("Using baseline predictions")
            val_enhanced['prediction'] = val_enhanced[targets[0]].mean()
            val_enhanced['prediction_weighted'] = val_enhanced[targets[0]].mean()
            results['feature_importance'] = pd.DataFrame({
                'feature': features[:min(10, len(features))],
                'importance': [1.0 / min(10, len(features))] * min(10, len(features))
            })
            default_metrics = {
                'mean_correlation': 0,
                'std_correlation': 1e-10,
                'sharpe_ratio': 0,
                'overall_correlation': 0,
                'feature_neutral_correlation': 0,
                'worst_era_correlation': 0,
                'best_era_correlation': 0
            }
            results['results_standard'] = default_metrics
            results['results_weighted'] = default_metrics.copy()

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        traceback.print_exc()
        if not results:
            results = {
                'error': str(e),
                'feature_groups': feature_groups or {"domain_0": features},
                'embedding': embedding,
                'cluster_labels': cluster_labels
            }

    print("AlphaFold-inspired pipeline completed")
    return results


def run_domains_only_pipeline():
    """
    Run just the domain extraction part of the pipeline.

    This function loads data, extracts feature domains, and saves both basic and detailed domain information.
    
    Returns:
        dict: Results dictionary containing domain information and paths to saved artifacts.
    """
    try:
        print("Loading data...")
        # Note: Replace load_data with the proper import or implementation if available.
        train_df, val_df, features, targets = load_data()
        if train_df is None:
            print("Failed to load data. Exiting.")
            return None

        # Use a subset of data for faster processing if dataset is large
        if len(train_df) > 50000:
            print(f"Using a subset of data for faster domain extraction (original size: {len(train_df)})")
            train_df = train_df.sample(50000, random_state=42)

        # Extract domains using the dedicated function
        results = extract_feature_domains_only(
            train_df,
            features,
            n_clusters=min(15, len(features) // 3),
            save_path='feature_domains_data.csv'
        )

        # Save feature list for reference
        try:
            feature_info = pd.DataFrame({'feature_name': features})
            feature_info.to_csv('feature_list.csv', index=False)
            print("Feature list saved to feature_list.csv")
        except Exception as feat_error:
            print(f"Error saving feature list: {feat_error}")

        return results

    except Exception as e:
        print(f"Error running domains-only pipeline: {e}")
        traceback.print_exc()
        return {'error': str(e)}


def extract_feature_domains_only(train_df, features, n_clusters=10,
                                 random_seed=42, save_path='feature_domains_data.csv'):
    """
    Extract and save feature domains without running the full pipeline.

    Parameters:
        train_df (DataFrame): Training data with features.
        features (list): List of feature names.
        n_clusters (int): Number of clusters to create.
        random_seed (int): Seed for reproducibility.
        save_path (str): Path to save the domain data CSV.

    Returns:
        dict: Dictionary with domain information, including paths to saved artifacts and visualizations.
    """
    print(f"Extracting feature domains from {len(features)} features...")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    try:
        # Identify feature domains
        feature_groups, embedding, cluster_labels, _ = identify_feature_domains(
            train_df, features, n_clusters=n_clusters, random_state=random_seed
        )
        num_domains = len(feature_groups)
        print(f"Identified {num_domains} feature domains")

        try:
            # Save domain data to CSV
            saved_path = save_feature_domains_data(
                feature_groups, embedding, cluster_labels, features,
                output_path=save_path
            )
            print(f"Feature domain data saved to: {saved_path}")

            # Save detailed domain information
            try:
                domain_details = []
                unique_domains = np.unique(cluster_labels)
                for domain_id in unique_domains:
                    domain_features = [f for i, f in enumerate(features) if cluster_labels[i] == domain_id]
                    center_x = np.mean(embedding[cluster_labels == domain_id, 0]) if embedding is not None else np.nan
                    center_y = np.mean(embedding[cluster_labels == domain_id, 1]) if embedding is not None else np.nan
                    domain_details.append({
                        'domain_id': domain_id,
                        'feature_count': len(domain_features),
                        'features': ', '.join(domain_features),
                        'center_x': center_x,
                        'center_y': center_y
                    })
                domain_details_df = pd.DataFrame(domain_details)
                details_path = save_path.replace('.csv', '_details.csv')
                domain_details_df.to_csv(details_path, index=False)
                print(f"Domain details saved to: {details_path}")
            except Exception as details_error:
                print(f"Error saving domain details: {details_error}")

            # Create and save visualizations
            try:
                plt.figure(figsize=(14, 12))
                domain_plot = visualize_feature_domains(embedding, cluster_labels, features)
                plot_path = save_path.replace('.csv', '_plot.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Domain visualization saved to: {plot_path}")

                # Create interactive visualization if possible
                try:
                    interactive_path = save_path.replace('.csv', '_interactive.html')
                    create_interactive_domain_visualization(embedding, cluster_labels, features, save_path=interactive_path)
                    print(f"Interactive visualization saved to: {interactive_path}")
                except ImportError:
                    print("Plotly not available, skipping interactive visualization")

                # Save domain similarity heatmap
                try:
                    plt.figure(figsize=(14, 12))
                    heatmap_path = save_path.replace('.csv', '_heatmap.png')
                    visualize_domain_heatmap(embedding, cluster_labels, features, save_path=heatmap_path)
                    print("Domain similarity heatmap saved")
                except Exception as heatmap_error:
                    print(f"Error creating heatmap: {heatmap_error}")
            except Exception as viz_error:
                print(f"Visualization failed: {viz_error}")

            # Analyze domain relationships if enough data is available
            try:
                if len(train_df) >= 1000:
                    relationships_path = save_path.replace('.csv', '_relationships.csv')
                    analyze_domain_relationships(train_df, features, cluster_labels, output_path=relationships_path)
                    print("Domain relationship analysis saved")
            except Exception as rel_error:
                print(f"Error analyzing domain relationships: {rel_error}")

            return {
                'feature_groups': feature_groups,
                'num_domains': num_domains,
                'domains_saved_path': saved_path,
                'embedding_shape': embedding.shape if embedding is not None else None,
                'cluster_labels': cluster_labels,
                'visualizations': {
                    'clustering': plot_path,
                    'interactive': interactive_path if 'interactive_path' in locals() else None,
                    'heatmap': heatmap_path if 'heatmap_path' in locals() else None
                }
            }
        except Exception as save_error:
            print(f"Error saving domain data: {save_error}")
            return {
                'feature_groups': feature_groups,
                'num_domains': num_domains,
                'error': str(save_error)
            }
    except Exception as e:
        print(f"Domain extraction failed: {e}")
        traceback.print_exc()
        return {'error': str(e)}
