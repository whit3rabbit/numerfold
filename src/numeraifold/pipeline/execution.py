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
import umap
import json
import os
import gc
from lightgbm import LGBMRegressor
import traceback
from typing import Optional, Dict, List
import pyarrow.parquet as pq
import seaborn as sns
from collections import defaultdict
from feature_engine.selection import SmartCorrelatedSelection

# Sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.manifold import TSNE

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
from numeraifold.domains.evaluate import evaluate_domain_performance

# Feature engineering and stability analysis
from numeraifold.features.engineering import generate_alphafold_features
from numeraifold.features.stability import calculate_feature_stability

# Utilities for saving/loading artifacts
from numeraifold.utils.artifacts import save_model_artifacts, save_feature_domains_data, load_and_analyze_domains

# Serialize domain results
from numeraifold.utils.serialize import make_json_serializable, save_results_to_json

# Load/save domain models
from numeraifold.utils.domain import save_model_and_domains, load_model_and_domains

# Logging
from numeraifold.utils.logging import log_memory_usage

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
                           skip_phase1=False, reuse_model=False, model_path=None):
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
        reuse_model (bool): If True, try to load a previously saved model.
        model_path (str): Path to the saved model directory.

    Returns:
        dict: Results dictionary containing domain data, trained model, evaluation metrics, etc.
    """
    print("Starting AlphaFold-inspired Numerai pipeline...")

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    results = {}
    embedding = None
    cluster_labels = None
    feature_groups = None
    trained_model = None
    model_loaded = False

    try:
        # First check if we should reuse a previously saved model
        if reuse_model and model_path is not None:
            print(f"Attempting to load saved model from {model_path}")
            try:
                loaded_data = load_model_and_domains(model_path)
                if 'model' in loaded_data and loaded_data['model'] is not None:
                    print("Successfully loaded model")
                    trained_model = loaded_data['model']
                    model_loaded = True
                    results['trained_model'] = trained_model
                    results['model_loaded'] = True
                    
                    # If we have feature groups, load those too
                    if 'feature_groups' in loaded_data and loaded_data['feature_groups']:
                        feature_groups = loaded_data['feature_groups']
                        results['feature_groups'] = feature_groups
                        print(f"Loaded {len(feature_groups)} feature groups")
                        
                        # If we loaded feature groups, we can skip Phase 1
                        skip_phase1 = True
                    
                    # If we have pruned features, use those
                    if 'pruned_features' in loaded_data and loaded_data['pruned_features']:
                        pruned_features = loaded_data['pruned_features']
                        results['pruned_features'] = pruned_features
                        print(f"Loaded {len(pruned_features)} pruned features")
                        
                        # Use these as our features going forward
                        features = [f for f in pruned_features if f in train_df.columns and f in val_df.columns]
                        print(f"Using {len(features)} loaded features")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Will train new model instead")

        # Convert PyArrow types in train_df and val_df to standard types
        print("Converting data types to ensure compatibility...")
        # Helper function to convert numeric columns to float32
        def convert_numeric_to_float32(df, cols_to_convert):
            for col in cols_to_convert:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype('float32')
            return df
            
        # Convert features in both dataframes
        train_df = convert_numeric_to_float32(train_df, features)
        val_df = convert_numeric_to_float32(val_df, features)
        
        # Convert target columns
        if isinstance(targets, list):
            train_df = convert_numeric_to_float32(train_df, targets)
            val_df = convert_numeric_to_float32(val_df, targets)
        else:
            if targets in train_df.columns:
                train_df[targets] = train_df[targets].astype('float32')
            if targets in val_df.columns:
                val_df[targets] = val_df[targets].astype('float32')

        # ----- Phase 1: Feature Domain Identification -----
        print("----- Phase 1: Feature Domain Identification -----")
        if force_phase1:
            print("Force Phase 1 enabled. Running domain identification regardless of cached data.")
            skip_phase1 = False
            
        if skip_phase1 and not force_phase1:
            print(f"Skipping Phase 1. Loading domain data from: {domains_save_path}")
            # Use the utility to load domain data
            phase1_data = load_and_analyze_domains(domains_save_path)
            
            if phase1_data is not None and isinstance(phase1_data, dict):
                if 'create_feature_groups' in phase1_data and callable(phase1_data['create_feature_groups']):
                    feature_groups = phase1_data['create_feature_groups']()
                    embedding = phase1_data.get('data', {}).get('embedding', None)
                    if 'data' in phase1_data and isinstance(phase1_data['data'], pd.DataFrame):
                        cluster_labels = phase1_data['data']['domain_id'].values if 'domain_id' in phase1_data['data'].columns else None
                    
                    if feature_groups:
                        print(f"Successfully loaded {len(feature_groups)} feature groups from saved data.")
                        results.update(phase1_data)
                    else:
                        print("Failed to create feature groups from saved data. Proceeding with Phase 1.")
                        skip_phase1 = False
                else:
                    print("Invalid domain data format. Proceeding with Phase 1.")
                    skip_phase1 = False
            else:
                print(f"Error loading domain data from {domains_save_path}. Proceeding with Phase 1.")
                skip_phase1 = False

        if not skip_phase1 or feature_groups is None:
            print("Running Phase 1: Feature Domain Identification")
            # Perform feature domain identification
            feature_groups, embedding, cluster_labels, _ = identify_feature_domains(
                train_df, features, n_clusters=n_clusters, random_state=random_seed
            )
            
            if not feature_groups:
                print("Warning: No feature groups created. Using single group.")
                feature_groups = {"domain_0": features}
            
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

            # Call the follow-up pipeline to refine domains
            print("Running follow-up domains pipeline to refine and prune features...")
            followup_results = follow_up_domains_pipeline(
                train_df=train_df,
                val_df=val_df,
                feature_groups=feature_groups,
                main_target="target",
                domain_score_threshold=0.01,
                correlation_threshold=0.95
            )
            
            # Make the followup results JSON serializable
            followup_results = make_json_serializable(followup_results)
            results['followup_results'] = followup_results
            
            # Save followup results to a separate file for inspection
            try:
                with open('followup_results.json', 'w') as f:
                    json.dump(followup_results, f, indent=4)
                print("Follow-up results saved to followup_results.json")
            except Exception as e:
                print(f"Warning: Failed to save followup results: {e}")
                
            refined_features = followup_results.get('pruned_features', features)
            print(f"Refined features count: {len(refined_features)}")

            features = refined_features # Set refined features

        else:
            # If Phase 1 is completely skipped
            print("Phase 1 fully skipped.")

        # ----- Phase 2: Model Architecture -----
        print("----- Phase 2: Model Architecture -----")

        # Skip model creation if we loaded one successfully
        if not model_loaded:
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
                # Convert data to float32 to avoid PyArrow type issues
                X_train = np.array(train_df[features].fillna(0).astype('float32').values, dtype=np.float32)
                X_val = np.array(val_df[features].fillna(0).astype('float32').values, dtype=np.float32)

                y_train = np.array(train_df[targets[0]].fillna(0).astype('float32').values, dtype=np.float32).reshape(-1, 1)
                y_val = np.array(val_df[targets[0]].fillna(0).astype('float32').values, dtype=np.float32).reshape(-1, 1)

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
                    # Try again with explicit float conversion for safety
                    X_train_np = np.array(train_df[features].fillna(0).astype('float32').values, dtype=np.float32)
                    y_train_np = np.array(train_df[targets[0]].fillna(0).astype('float32').values, dtype=np.float32).reshape(-1, 1)
                    X_val_np = np.array(val_df[features].fillna(0).astype('float32').values, dtype=np.float32)
                    y_val_np = np.array(val_df[targets[0]].fillna(0).astype('float32').values, dtype=np.float32).reshape(-1, 1)

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
                    X_train_min = torch.tensor(train_df[feature].fillna(0).astype('float32').values, dtype=torch.float32).view(-1, 1)
                    y_train_min = torch.tensor(train_df[targets[0]].fillna(0).astype('float32').values, dtype=torch.float32).view(-1, 1)
                    X_val_min = torch.tensor(val_df[feature].fillna(0).astype('float32').values, dtype=torch.float32).view(-1, 1)
                    y_val_min = torch.tensor(val_df[targets[0]].fillna(0).astype('float32').values, dtype=torch.float32).view(-1, 1)

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
        else:
            print("Using loaded model, skipping training")
            
        # Save the model and domain data if requested
        if save_model and 'trained_model' in results and results['trained_model'] is not None:
            model_save_dir = os.path.join(base_path, 'saved_models')
            try:
                # Get domain scores from followup results if available
                domain_scores = {}
                if 'followup_results' in results and 'domain_scores' in results['followup_results']:
                    domain_scores = results['followup_results']['domain_scores']
                
                # Get pruned features if available
                pruned_features = features
                if 'followup_results' in results and 'pruned_features' in results['followup_results']:
                    pruned_features = results['followup_results']['pruned_features']
                
                saved_paths = save_model_and_domains(
                    model=results['trained_model'],
                    feature_groups=feature_groups,
                    domain_scores=domain_scores,
                    pruned_features=pruned_features,
                    save_dir=model_save_dir
                )
                results['saved_model_paths'] = saved_paths
                print(f"Model and data saved to {model_save_dir}")
            except Exception as e:
                print(f"Error saving model: {e}")

        # If only domain data is needed and no model is trained, exit early
        if results.get('domains_saved_path') and 'trained_model' not in results:
            print("Domain data saved. Skipping remaining steps as requested.")
            return results

# ----- Phase 3: Feature Generation -----
        print("----- Phase 3: Feature Generation -----")
        if trained_model is not None:
            try:
                print("Generating AlphaFold-inspired features...")
                # Make sure to convert types before passing to generate_alphafold_features
                train_df_float = train_df.copy()
                val_df_float = val_df.copy()
                
                # Convert feature columns to float32
                for col in features:
                    if col in train_df_float.columns and pd.api.types.is_numeric_dtype(train_df_float[col]):
                        train_df_float[col] = train_df_float[col].astype('float32')
                    if col in val_df_float.columns and pd.api.types.is_numeric_dtype(val_df_float[col]):
                        val_df_float[col] = val_df_float[col].astype('float32')
                
                train_features_df, val_features_df = generate_alphafold_features(
                    train_df_float, val_df_float, trained_model, features,
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
                target_col = targets[0] if isinstance(targets, list) else targets
                train_enhanced['prediction'] = train_enhanced[target_col].astype('float32').mean()
                train_enhanced['af_high_confidence'] = 0
                val_enhanced['af_confidence'] = 0.5
                val_enhanced['prediction'] = val_enhanced[target_col].astype('float32').mean()
                val_enhanced['af_high_confidence'] = 0
        else:
            print("Skipping feature generation due to model training failure")
            train_enhanced = train_df.copy()
            val_enhanced = val_df.copy()
            target_col = targets[0] if isinstance(targets, list) else targets
            train_enhanced['af_confidence'] = 0.5
            train_enhanced['prediction'] = train_enhanced[target_col].astype('float32').mean()
            train_enhanced['af_high_confidence'] = 0
            val_enhanced['af_confidence'] = 0.5
            val_enhanced['prediction'] = val_enhanced[target_col].astype('float32').mean()
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

            # Convert all feature columns to float32 for evaluation
            val_enhanced_float = val_enhanced.copy()
            for col in all_features:
                if col in val_enhanced_float.columns and pd.api.types.is_numeric_dtype(val_enhanced_float[col]):
                    val_enhanced_float[col] = val_enhanced_float[col].astype('float32')

            # Run evaluation on the final model
            eval_results = run_final_evaluation(
                val_df=val_enhanced_float,
                model=trained_model,
                val_loader=val_loader,
                targets=targets,
                era_col='era',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            if eval_results is not None:
                # Make evaluation results JSON serializable
                eval_results = make_json_serializable(eval_results)
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
            target_col = targets[0] if isinstance(targets, list) else targets
            val_enhanced['prediction'] = val_enhanced[target_col].astype('float32').mean()
            val_enhanced['prediction_weighted'] = val_enhanced[target_col].astype('float32').mean()
            
            # Create a serializable feature importance dataframe
            feature_importance_dict = {
                'feature': features[:min(10, len(features))],
                'importance': [1.0 / min(10, len(features))] * min(10, len(features))
            }
            results['feature_importance'] = feature_importance_dict
            
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

def chunked_data_loader(
    data_version: str,
    feature_set: str,
    chunk_size: int = 10000,
    target_params: Optional[Dict] = None,
    convert_dtypes: bool = True  # Add parameter to control type conversion
):
    """
    Generator to load data in chunks with memory-efficient type conversion.
    
    Args:
        data_version: Version of dataset
        feature_set: Size of feature set
        chunk_size: Size of chunks to load
        target_params: Dictionary containing target configuration:
            - main_target: Primary target column
            - num_aux_targets: Number of auxiliary targets
        convert_dtypes: If True, converts numeric columns to float32
    """
    if target_params is None:
        target_params = {
            'main_target': 'target',
            'num_aux_targets': 3
        }
    
    file_path = f"{data_version}/train.parquet"
    
    # Read schema and feature metadata
    schema = pq.read_schema(file_path)
    all_columns = schema.names
    
    # Get features
    with open(f"{data_version}/features.json") as f:
        import json
        feature_metadata = json.load(f)
    features = feature_metadata["feature_sets"][feature_set]
    
    # Get targets
    available_targets = [col for col in all_columns if col.startswith('target')]
    main_target = target_params.get('main_target', 'target')
    if main_target not in available_targets:
        print(f"Warning: {main_target} not found in dataset. Using first available target.")
        main_target = available_targets[0] if available_targets else None
    
    num_aux_targets = target_params.get('num_aux_targets', 3)
    aux_targets = [t for t in available_targets if t != main_target][:num_aux_targets]
    all_targets = [main_target] + aux_targets if main_target else aux_targets
    
    if not all_targets:
        print("Warning: No target columns found.")
    
    # Create ParquetFile object
    parquet_file = pq.ParquetFile(file_path)
    
    # Prepare optimized schema for conversion if needed
    if convert_dtypes:
        columns_to_load = ["era"] + features + all_targets
        modified_fields = []
        for field in schema:
            if field.name in columns_to_load:
                if field.name == 'era':
                    modified_fields.append(field)
                else:
                    # Check if the field type is numeric
                    if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type):
                        modified_fields.append(pa.field(field.name, pa.float32()))
                    else:
                        modified_fields.append(field)
        optimized_schema = pa.schema(modified_fields)
    else:
        optimized_schema = None
    
    # Read in chunks
    for row_group in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(row_group, columns=["era"] + features + all_targets)
        
        # Apply type conversion if needed
        if convert_dtypes:
            for col in batch.column_names:
                if col == 'era':
                    continue
                field_type = batch.schema.field(col).type
                if (pa.types.is_integer(field_type) or 
                    pa.types.is_floating(field_type) or 
                    pa.types.is_decimal(field_type)):
                    batch = batch.set_column(
                        batch.column_names.index(col),
                        col,
                        batch[col].cast(pa.float32())
                    )
        
        # Convert to pandas without dtype_backend
        df = batch.to_pandas()
        
        # Manual conversion to float32 if needed and not already done at PyArrow level
        if convert_dtypes:
            numeric_cols = df.select_dtypes(include=np.number).columns.difference(['era'])
            if not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].astype('float32')
        
        yield df, features, all_targets

def run_domains_only_pipeline(
    data_version: str = "v5.0",
    feature_set: str = "medium",
    main_target: str = "target",
    num_aux_targets: int = 3,
    aux_targets: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    n_clusters: Optional[int] = None,  # if None, dynamic clustering will determine optimal clusters
    save_path: str = 'feature_domains_data.csv',
    chunk_size: int = 10000,
    use_incremental: bool = True,
    skip_visualizations: bool = False,
    random_seed: int = 42,
    convert_dtypes: bool = True  # Add parameter to control type conversion
) -> Dict:
    print(f"Starting enhanced domain pipeline with {feature_set} feature set")
    log_memory_usage("Initial")
    results = {}

    try:
        # ---------------------------
        # 1. Data Loading & Preparation - Modified to use optimized loading
        # ---------------------------
        print("Loading data...")
        if aux_targets is not None:
            print(f"Using specified auxiliary targets: {aux_targets}")
            # Use the optimized loader that doesn't use dtype_backend
            train_df, val_df, features, all_targets = load_data(
                data_version=data_version,
                feature_set=feature_set,
                main_target=main_target,
                aux_targets=aux_targets,  # Pass the aux_targets directly
                num_aux_targets=0,  # Set to 0 to prevent random selection
                convert_dtypes=convert_dtypes,  # Control type conversion
                chunk_size=chunk_size if use_incremental else None  # Only use chunking if incremental is enabled
            )
            
            # Check if all specified targets are in the loaded data
            available_targets = [col for col in train_df.columns if col.startswith('target')]
            print(f"Available targets: {available_targets}")
            missing_targets = [t for t in aux_targets if t not in train_df.columns]
            if missing_targets:
                raise ValueError(f"Specified auxiliary targets not found in data: {missing_targets}")
            
            all_targets = [main_target] + aux_targets  # Use the specified targets
        else:
            print(f"Selecting {num_aux_targets} random auxiliary targets")
            # Use the optimized loader that doesn't use dtype_backend
            train_df, val_df, features, all_targets = load_data(
                data_version=data_version,
                feature_set=feature_set,
                main_target=main_target,
                num_aux_targets=num_aux_targets,
                convert_dtypes=convert_dtypes,  # Control type conversion
                chunk_size=chunk_size if use_incremental else None  # Only use chunking if incremental is enabled
            )
        
        if train_df is None or val_df is None:
            raise ValueError("Failed to load data")
            
        print(f"Loaded data with {len(features)} features")
        print(f"Using targets: {all_targets}")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        # Apply sampling if specified
        if sample_size is not None and sample_size < len(train_df):
            print(f"Sampling {sample_size} rows from training data")
            train_df = train_df.sample(n=sample_size, random_state=random_seed)
        
        # Add explicit garbage collection after loading
        gc.collect()
        log_memory_usage("After data loading")
        
        # ---------------------------
        # 2. Compute Feature-Target Correlations
        # ---------------------------
        print("Computing feature-target correlations...")
        correlations = []
        
        # Use memory-efficient batch processing for large datasets
        if len(features) > 1000 and use_incremental:
            print("Using memory-efficient correlation calculation...")
            batch_size = 500  # Process features in smaller batches
            
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_correlations = []
                
                for feature in batch_features:
                    feature_corrs = []
                    for target in all_targets:
                        # Handle NaNs by filling with 0
                        feature_data = train_df[feature].fillna(0).values
                        target_data = train_df[target].fillna(0).values
                        corr = np.corrcoef(feature_data, target_data)[0, 1]
                        feature_corrs.append(corr)
                    batch_correlations.append(feature_corrs)
                
                correlations.extend(batch_correlations)
                
                # Clean up batch memory
                del batch_correlations
                gc.collect()
        else:
            # Standard processing for smaller datasets
            for feature in features:
                feature_corrs = []
                for target in all_targets:
                    # Handle NaNs by filling with 0
                    feature_data = train_df[feature].fillna(0).values
                    target_data = train_df[target].fillna(0).values
                    corr = np.corrcoef(feature_data, target_data)[0, 1]
                    feature_corrs.append(corr)
                correlations.append(feature_corrs)
        
        correlations = np.array(correlations)
        print(f"Correlation matrix shape: {correlations.shape}")
        
        # ---------------------------
        # 3. Enhanced Embeddings using t-SNE
        # ---------------------------
        print("Applying t-SNE for enhanced feature embeddings...")
        tsne = TSNE(n_components=3, perplexity=30, random_state=random_seed)
        feature_embeddings = tsne.fit_transform(correlations)
        print(f"Feature embeddings shape: {feature_embeddings.shape}")
        
        # Add garbage collection after feature embedding
        del correlations
        gc.collect()
        log_memory_usage("After feature embedding")

        # ---------------------------
        # 4. Dynamic Cluster Determination
        # ---------------------------
        if n_clusters is None:
            print("Determining optimal number of clusters dynamically using silhouette scores...")
            silhouette_scores = []
            k_range = range(10, 30)
            for k in k_range:
                clusterer = KMeans(n_clusters=k, random_state=random_seed)
                labels = clusterer.fit_predict(feature_embeddings)
                score = silhouette_score(feature_embeddings, labels)
                silhouette_scores.append(score)
                print(f"Silhouette score for k={k}: {score:.4f}")
            optimal_k = k_range[np.argmax(silhouette_scores)]
            n_clusters = optimal_k
            print(f"Optimal number of clusters determined: {n_clusters}")
        else:
            print(f"Using user-specified number of clusters: {n_clusters}")

        # ---------------------------
        # 5. Clustering Features using KMeans
        # ---------------------------
        print(f"Clustering features into {n_clusters} domains...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_seed)
        cluster_labels = clusterer.fit_predict(feature_embeddings)

        # Create feature groups (domains)
        feature_groups = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            domain_name = f"domain_{label}"
            feature_groups[domain_name].append(features[idx])
        feature_groups = dict(feature_groups)

        # ---------------------------
        # 6. Save Domain Data
        # ---------------------------
        results['feature_groups'] = feature_groups
        results['num_domains'] = len(feature_groups)
        results['targets_used'] = all_targets

        try:
            print("Saving domain data...")
            domain_data = []
            for idx, feature in enumerate(features):
                domain_data.append({
                    'feature': feature,
                    'domain_id': int(cluster_labels[idx]),
                    'domain_name': f"domain_{int(cluster_labels[idx])}",
                    **{f'dimension_{i+1}': emb for i, emb in enumerate(feature_embeddings[idx])}
                })
            domains_df = pd.DataFrame(domain_data)
            domains_df.to_csv(save_path, index=False)
            results['domains_saved_path'] = save_path
            print(f"Domain data saved to: {save_path}")
        except Exception as save_error:
            print(f"Error saving domain data: {save_error}")
            results['save_error'] = str(save_error)
        
        # Clean up memory before cluster validation
        del feature_embeddings
        gc.collect()
        log_memory_usage("After clustering")
        
        # ---------------------------
        # 7. Cluster Validation: Intra-domain Correlation Analysis
        # ---------------------------
        print("Computing intra-domain correlation matrix...")
        domain_corr_matrix = np.zeros((n_clusters, len(all_targets)))
        for domain_id in range(n_clusters):
            domain_features = [features[i] for i, label in enumerate(cluster_labels) if label == domain_id]
            for j, target in enumerate(all_targets):
                corrs = [
                    np.corrcoef(train_df[feat].fillna(0).values, train_df[target].fillna(0).values)[0, 1]
                    for feat in domain_features
                ]
                domain_corr_matrix[domain_id, j] = np.mean(corrs)
        results['domain_validation'] = domain_corr_matrix.tolist()
        
        # ---------------------------
        # 8. Additional Visualizations
        # ---------------------------
        if not skip_visualizations:
            try:
                print("Generating additional visualizations using UMAP...")
                reducer = umap.UMAP(random_state=random_seed)
                vis_embeddings = reducer.fit_transform(clusterer.cluster_centers_)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(vis_embeddings[:, 0], vis_embeddings[:, 1],
                                      c=range(n_clusters), cmap='tab20', s=100)
                plt.colorbar(scatter)
                plt.title("Feature Domains Visualization (UMAP)")
                
                plot_path = save_path.replace('.csv', '_plot.png')
                plt.savefig(plot_path, dpi=120, bbox_inches='tight')
                plt.close()
                results['plot_path'] = plot_path
                print("Visualization generated successfully")
            except Exception as viz_error:
                print(f"Error generating visualizations: {viz_error}")
                results['visualization_error'] = str(viz_error)
        
        # Final memory clean-up
        gc.collect()
        log_memory_usage("Final")
        
        return results

    except Exception as e:
        print(f"Pipeline error: {e}")
        traceback.print_exc()
        return {'error': str(e)}

def memory_efficient_feature_pruning(
    df, features, target_col, correlation_threshold=0.95, batch_size=100
):
    """
    Memory-efficient implementation of feature correlation pruning.
    Processes features in batches to avoid loading full correlation matrix.
    
    Args:
        df: DataFrame containing features
        features: List of feature names
        target_col: Target column name
        correlation_threshold: Threshold above which features are considered highly correlated
        batch_size: Size of feature batches to process
        
    Returns:
        List of selected features after pruning
    """
    print(f"Starting memory-efficient feature pruning of {len(features)} features...")
    
    # Always keep features with highest correlation to target
    target_corrs = {}
    remaining_features = features.copy()
    
    # Process in batches to avoid memory issues
    for i in range(0, len(features), batch_size):
        batch = features[i:min(i+batch_size, len(features))]
        batch_df = df[batch + [target_col]].astype('float32')
        
        # Calculate correlation to target
        batch_corrs = batch_df[batch].corrwith(batch_df[target_col]).abs()
        
        # Update target correlations
        for feat, corr in batch_corrs.items():
            target_corrs[feat] = corr
            
        # Clean up
        del batch_df
        del batch_corrs
        gc.collect()
    
    # Sort features by correlation to target
    sorted_features = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize selected features with the one most correlated to target
    selected = [sorted_features[0][0]]
    remaining_features.remove(selected[0])
    
    print(f"Starting with feature most correlated to target: {selected[0]}")
    
    # Process in batches of features to check correlation against selected features
    while remaining_features:
        # Take a batch of unprocessed features
        batch_size_adjusted = min(batch_size, len(remaining_features))
        if batch_size_adjusted == 0:
            break
            
        # Sort remaining by target correlation and take top batch
        remaining_sorted = sorted(
            [(f, target_corrs[f]) for f in remaining_features],
            key=lambda x: x[1],
            reverse=True
        )
        current_batch = [f[0] for f in remaining_sorted[:batch_size_adjusted]]
        
        # Remove this batch from remaining
        for feat in current_batch:
            remaining_features.remove(feat)
        
        # Check correlation of batch with already selected features
        to_select_from_batch = []
        for feature in current_batch:
            keep = True
            
            # Process in sub-batches of selected features
            for i in range(0, len(selected), batch_size):
                selected_batch = selected[i:min(i+batch_size, len(selected))]
                
                # Get correlation between this feature and selected batch
                columns_to_check = selected_batch + [feature]
                corr_df = df[columns_to_check].astype('float32').corr().abs()
                
                # Check if this feature is highly correlated with any already selected feature
                for sel_feat in selected_batch:
                    if corr_df.loc[feature, sel_feat] > correlation_threshold:
                        keep = False
                        break
                
                # No need to check other batches if we're discarding this feature
                if not keep:
                    break
                    
                # Clean up
                del corr_df
                gc.collect()
            
            if keep:
                to_select_from_batch.append(feature)
        
        # Add batch's selected features to the final selection
        selected.extend(to_select_from_batch)
        print(f"Selected {len(to_select_from_batch)} features from batch, {len(selected)} total")
    
    print(f"Final selected features: {len(selected)}/{len(features)}")
    return selected

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

def follow_up_domains_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: dict,
    main_target: str,
    domain_score_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    memory_efficient: bool = True,  # Default to memory efficient mode
    batch_size: int = 100,
    convert_dtypes: bool = True  # Add parameter to control type conversion
):
    """
    Follow-up pipeline that:
      1) Evaluates each domain's performance using correlation metrics
      2) Filters out low-performing domains
      3) Prunes highly correlated features within the kept domains
      4) Trains a final global regression model on the pruned features

    Args:
        train_df (pd.DataFrame): Training data (features + target).
        val_df   (pd.DataFrame): Validation data (features + target).
        feature_groups (dict): { 'domain_0': [feat1, feat2, ...], ... }
        main_target (str): Name of the target column in train_df/val_df.
        domain_score_threshold (float): Minimum correlation threshold to keep a domain.
        correlation_threshold (float): Threshold for feature correlation pruning.
        memory_efficient (bool): If True, use memory efficient processing
        batch_size (int): Number of features to process at once in memory-efficient mode
        convert_dtypes (bool): If True, converts data to float32 for memory efficiency

    Returns:
        dict: Contains domain_scores, kept_domains, final_model_score, pruned_features, etc.
    """
    # -------------------------------------------------
    # 1. Evaluate domain performance using correlation
    # -------------------------------------------------
    print("Evaluating domain performance...")
    domain_scores = {}
    
    # Always ensure target column is float32 for consistent correlation calculations
    if convert_dtypes:
        target_series = train_df[main_target].astype('float32')
    else:
        target_series = train_df[main_target]
    
    # If memory_efficient is True, process domains in batches
    if memory_efficient:
        for domain_name, feats in feature_groups.items():
            # Handle empty or tiny feature groups
            if len(feats) < 5:
                print(f"  {domain_name}: Too few features ({len(feats)}), skipping")
                domain_scores[domain_name] = 0.0
                continue
                
            # Process features in batches
            try:
                correlations = []
                for i in range(0, len(feats), batch_size):
                    batch_feats = feats[i:i+batch_size]
                    print(f"  {domain_name}: Processing batch {i//batch_size + 1}/{(len(feats)+batch_size-1)//batch_size} ({len(batch_feats)} features)")
                    
                    # Convert features to float32 and calculate correlation
                    if convert_dtypes:
                        features_batch = train_df[batch_feats].astype('float32')
                    else:
                        features_batch = train_df[batch_feats]
                        
                    batch_corrs = features_batch.corrwith(target_series).abs()
                    correlations.extend(batch_corrs.tolist())
                    
                    # Clean up to free memory
                    del features_batch
                    del batch_corrs
                    gc.collect()
                
                # Calculate mean correlation for the domain
                mean_corr = sum(correlations) / len(correlations)
                domain_scores[domain_name] = float(mean_corr)
                
                # Clean up
                del correlations
                gc.collect()
                
            except Exception as e:
                print(f"  Error evaluating {domain_name}: {e}")
                domain_scores[domain_name] = 0.0
    else:
        # Standard processing without memory optimization
        for domain_name, feats in feature_groups.items():
            # Handle empty or tiny feature groups
            if len(feats) < 5:
                print(f"  {domain_name}: Too few features ({len(feats)}), skipping")
                domain_scores[domain_name] = 0.0
                continue
                
            # Calculate mean absolute correlation with target
            try:
                # Convert both features and target to float32 for consistency
                if convert_dtypes:
                    features_float = train_df[feats].astype('float32')
                    target_float = train_df[main_target].astype('float32')
                else:
                    features_float = train_df[feats]
                    target_float = train_df[main_target]
                
                correlations = features_float.corrwith(target_float).abs()
                mean_corr = correlations.mean()
                domain_scores[domain_name] = float(mean_corr)
                
                # Clean up
                del features_float
                del target_float
                del correlations
                gc.collect()
                
            except Exception as e:
                print(f"  Error evaluating {domain_name}: {e}")
                domain_scores[domain_name] = 0.0
    
    print("Domain performance scores (correlation with target):")
    for d, s in domain_scores.items():
        print(f"  {d}: {s:.4f}")

    # -------------------------------------------------
    # 2. Filter out low-performing domains
    # -------------------------------------------------
    print(f"\nFiltering domains with correlation >= {domain_score_threshold}...")
    kept_domains = [d for d, s in domain_scores.items() if s >= domain_score_threshold]
    print(f"Kept domains: {len(kept_domains)}/{len(domain_scores)}")
    
    # Handle the case where no domains meet the threshold
    if not kept_domains:
        print("Warning: No domains met threshold. Keeping top 3 domains instead.")
        kept_domains = sorted(domain_scores.keys(), key=lambda d: domain_scores[d], reverse=True)[:3]
        
    # Gather all features from kept domains
    kept_features = []
    for d in kept_domains:
        kept_features.extend(feature_groups[d])
    kept_features = list(set(kept_features))  # Remove duplicates
    print(f"Total features from kept domains: {len(kept_features)}")

    # -------------------------------------------------
    # 3. Correlation pruning within kept features
    # -------------------------------------------------
    print(f"\nPruning correlated features (threshold={correlation_threshold})...")
    
    # Use memory-efficient pruning for larger feature sets
    if memory_efficient or len(kept_features) > 1000:
        print("Using memory-efficient correlation pruning...")
        pruned_features = memory_efficient_feature_pruning(
            train_df, kept_features, main_target, 
            correlation_threshold=correlation_threshold, 
            batch_size=batch_size
        )
        print(f"Reduced from {len(kept_features)} to {len(pruned_features)} features after pruning")
    else:
        try:
            # Create a copy of required columns with proper types
            if convert_dtypes:
                columns_to_use = kept_features + [main_target]
                train_subset = train_df[columns_to_use].copy().astype('float32')
            else:
                columns_to_use = kept_features + [main_target]
                train_subset = train_df[columns_to_use].copy()
                
            # Calculate correlation matrix
            corr_matrix = train_subset.corr().abs()
            
            # Initialize with features sorted by correlation to target
            target_corrs = corr_matrix[main_target].drop(main_target).sort_values(ascending=False)
            
            # Start with top feature
            selected_features = [target_corrs.index[0]]
            remaining_features = target_corrs.index[1:].tolist()
            
            # Iteratively add features that aren't highly correlated with already selected ones
            for feature in remaining_features:
                # Check against all selected features
                highly_correlated = False
                for selected in selected_features:
                    if corr_matrix.loc[feature, selected] > correlation_threshold:
                        highly_correlated = True
                        break
                
                if not highly_correlated:
                    selected_features.append(feature)
            
            pruned_features = selected_features
            print(f"Reduced from {len(kept_features)} to {len(pruned_features)} features after pruning")
            
            # Clean up
            del train_subset
            del corr_matrix
            gc.collect()
            
        except Exception as e:
            print(f"Error in correlation pruning: {e}")
            print("Falling back to memory-efficient pruning")
            pruned_features = memory_efficient_feature_pruning(
                train_df, kept_features, main_target, 
                correlation_threshold=correlation_threshold, 
                batch_size=batch_size
            )

    # -------------------------------------------------
    # 4. Train a final global regression model on pruned features
    # -------------------------------------------------
    final_score = 0.0
    feature_importance = None
    
    # If we have a lot of features, use a subset for the model
    if len(pruned_features) > 500:
        print(f"\nToo many features for model training ({len(pruned_features)}). Using top 500 features by correlation.")
        
        # Use memory-efficient feature selection
        feature_scores = []
        
        if convert_dtypes:
            target_series = train_df[main_target].astype('float32')
        else:
            target_series = train_df[main_target]
        
        for i in range(0, len(pruned_features), batch_size):
            batch_feats = pruned_features[i:i+batch_size]
            
            if convert_dtypes:
                features_batch = train_df[batch_feats].astype('float32')
            else:
                features_batch = train_df[batch_feats]
                
            batch_corrs = features_batch.corrwith(target_series).abs()
            
            for feat, corr in zip(batch_feats, batch_corrs):
                feature_scores.append((feat, corr))
            
            # Clean up
            del features_batch
            del batch_corrs
            gc.collect()
            
        # Sort by correlation and take top 500
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        pruned_features = [f[0] for f in sorted_features[:500]]
        
        # Clean up
        del feature_scores
        del sorted_features
        gc.collect()
    
    print("\nTraining final global regression model with pruned features...")
    try:
        # Check if pruned features exist in both datasets
        valid_features = [f for f in pruned_features if f in train_df.columns and f in val_df.columns]
        if len(valid_features) < len(pruned_features):
            print(f"Warning: {len(pruned_features) - len(valid_features)} features not found in both datasets")
            pruned_features = valid_features
            
        if len(pruned_features) == 0:
            raise ValueError("No valid features remaining after pruning")
            
        # Convert all feature data to float32 to avoid PyArrow type issues
        if convert_dtypes:
            X_train = train_df[pruned_features].astype('float32')
            y_train = train_df[main_target].astype('float32')
            X_val = val_df[pruned_features].astype('float32')
            y_val = val_df[main_target].astype('float32')
        else:
            X_train = train_df[pruned_features]
            y_train = train_df[main_target]
            X_val = val_df[pruned_features]
            y_val = val_df[main_target]
        
        # Train the model with optimized parameters for Numerai
        final_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            colsample_bytree=0.7,
            subsample=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        final_model.fit(
            X_train, 
            y_train,
            eval_metric='rmse'
        )
        
        # Generate predictions and calculate correlation
        val_preds = final_model.predict(X_val)
        final_score = np.corrcoef(y_val.values, val_preds)[0, 1]
        print(f"Final model correlation score: {final_score:.4f}")
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': pruned_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Clean up to free memory
        del X_train, y_train, X_val, y_val
        gc.collect()
        
    except Exception as e:
        print(f"Error in final model training: {e}")
        print("Returning partial results without final model")

    # -------------------------------------------------
    # 5. Train domain-specific models and create ensemble
    # -------------------------------------------------
    domain_models = {}
    domain_preds = {}
    ensemble_score = 0.0
    
    try:
        print("\nTraining domain-specific models for ensemble...")
        for domain in kept_domains:
            domain_features = [f for f in feature_groups[domain] if f in train_df.columns and f in val_df.columns]
            
            # Skip domains with too few features
            if len(domain_features) < 5:
                print(f"  Skipping {domain}: too few valid features ({len(domain_features)})")
                continue
                
            # Take top 100 features by correlation if too many
            if len(domain_features) > 100:
                if convert_dtypes:
                    domain_corrs = train_df[domain_features].astype('float32').corrwith(
                        train_df[main_target].astype('float32')).abs()
                else:
                    domain_corrs = train_df[domain_features].corrwith(train_df[main_target]).abs()
                    
                domain_features = domain_corrs.sort_values(ascending=False).index[:100].tolist()
            
            print(f"  Training model for {domain} with {len(domain_features)} features")
            
            # Convert data to float32 if needed
            if convert_dtypes:
                X_train_domain = train_df[domain_features].astype('float32')
                y_train_domain = train_df[main_target].astype('float32')
                X_val_domain = val_df[domain_features].astype('float32')
            else:
                X_train_domain = train_df[domain_features]
                y_train_domain = train_df[main_target]
                X_val_domain = val_df[domain_features]
            
            # Create a simpler model for domain-specific prediction
            domain_model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=5,
                colsample_bytree=0.8,
                random_state=42
            )
            
            domain_model.fit(X_train_domain, y_train_domain)
            
            # Store model and generate predictions
            domain_models[domain] = domain_model
            domain_preds[domain] = domain_model.predict(X_val_domain)
            
            # Clean up
            del X_train_domain, y_train_domain, X_val_domain
            gc.collect()
        
        # Create simple average ensemble
        if domain_preds:
            print("\nCreating ensemble predictions...")
            ensemble_predictions = np.zeros(len(val_df))
            for domain, preds in domain_preds.items():
                ensemble_predictions += preds
            ensemble_predictions /= len(domain_preds)
            
            # Calculate ensemble score
            ensemble_score = np.corrcoef(val_df[main_target].values, ensemble_predictions)[0, 1]
            print(f"Ensemble model correlation score: {ensemble_score:.4f}")
    
    except Exception as e:
        print(f"Error in ensemble creation: {e}")
        print("Returning results without ensemble")

    # -------------------------------------------------
    # 6. Return results
    # -------------------------------------------------
    # Always return serializable objects
    if feature_importance is not None:
        feature_importance_dict = {
            'features': feature_importance['feature'].tolist(),
            'importance': feature_importance['importance'].tolist()
        }
    else:
        feature_importance_dict = {'features': [], 'importance': []}
        
    results = {
        'domain_scores': domain_scores,
        'kept_domains': kept_domains,
        'pruned_features': pruned_features,
        'final_model_score': float(final_score),
        'ensemble_score': float(ensemble_score),
        'feature_importance': feature_importance_dict
    }
    return results