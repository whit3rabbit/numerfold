"""
execution.py

This module implements the main pipeline execution functions for NumerAIFold,
including the full pipeline (feature domain identification, model training,
feature generation, and final evaluation) and a domains-only pipeline.
"""

import json
import os
import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from tqdm import tqdm
import umap
import gc
import os
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
                domain_score_threshold=0.51,   # You can adjust this threshold
                correlation_threshold=0.95       # Adjust pruning threshold if needed
            )
            refined_features = followup_results.get('pruned_features', features)
            print(f"Refined features count: {len(refined_features)}")

            features = refined_features # Set refined features

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

# Improved chunked_data_loader function with proper float32 conversion
def chunked_data_loader(
    data_version: str,
    feature_set: str,
    chunk_size: int = 10000,
    target_params: Optional[Dict] = None
):
    """
    Generator to load data in chunks with proper float32 conversion.
    
    Args:
        data_version: Version of dataset
        feature_set: Size of feature set
        chunk_size: Size of chunks to load
        target_params: Dictionary containing target configuration:
            - main_target: Primary target column
            - num_aux_targets: Number of auxiliary targets
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
    main_target = target_params.get('main_target', 'target')
    available_targets = [col for col in all_columns if col.startswith('target')]
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
    
    # Read in chunks and convert to float32
    for i in range(0, parquet_file.metadata.num_rows, chunk_size):
        try:
            # Read chunk with appropriate columns
            chunk = pd.read_parquet(
                file_path,
                columns=["era"] + features + all_targets,
                skip_rows=i,
                num_rows=min(chunk_size, parquet_file.metadata.num_rows - i)
            )
            
            # Convert features and targets to float32
            for feature in features:
                chunk[feature] = chunk[feature].astype(np.float32)
            
            for target in all_targets:
                if target in chunk.columns:
                    chunk[target] = chunk[target].astype(np.float32)
            
            # Report memory usage for monitoring
            memory_usage = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"Chunk {i//chunk_size + 1} loaded: {len(chunk)} rows, {memory_usage:.2f} MB")
            
            yield chunk, features, all_targets
            
        except Exception as e:
            print(f"Error loading chunk starting at row {i}: {e}")
            traceback.print_exc()
            continue

def follow_up_domains_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: dict,
    main_target: str,
    domain_score_threshold: float = 0.01,  # Lower threshold for correlation
    correlation_threshold: float = 0.95
):
    """
    Follow-up pipeline with improved NaN handling and robust correlation calculation:
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

    Returns:
        dict: Contains domain_scores, kept_domains, final_model_score, pruned_features, etc.
    """
    print("Starting follow-up domains pipeline with improved NaN handling...")
    
    # Check if target column exists
    if main_target not in train_df.columns:
        print(f"Error: Target column '{main_target}' not found in training data")
        print(f"Available columns: {train_df.columns[:10]}...")
        return {
            'error': f"Target column '{main_target}' not found in training data",
            'domain_scores': {},
            'kept_domains': [],
            'pruned_features': []
        }
    
    # -------------------------------------------------
    # 0. Clean data and ensure proper types
    # -------------------------------------------------
    print("Cleaning data and ensuring proper types...")
    
    # Check for NaN values in target
    target_null_count = train_df[main_target].isnull().sum()
    if target_null_count > 0:
        print(f"Warning: Target column has {target_null_count} NaN values. Filling with mean.")
        target_mean = train_df[main_target].mean()
        train_df[main_target] = train_df[main_target].fillna(target_mean)
    
    # Get all features from all domains
    all_features = []
    for feats in feature_groups.values():
        all_features.extend(feats)
    all_features = list(set(all_features))  # Remove duplicates
    
    # Check which features exist in the data
    missing_features = [f for f in all_features if f not in train_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features from domains not found in training data")
        if len(missing_features) <= 10:
            print(f"Missing features: {missing_features}")
        else:
            print(f"First 10 missing features: {missing_features[:10]}...")
    
    valid_features = [f for f in all_features if f in train_df.columns]
    print(f"Found {len(valid_features)}/{len(all_features)} valid features in training data")
    
    # Fill NaN values in features with 0
    na_counts = train_df[valid_features].isna().sum()
    features_with_na = na_counts[na_counts > 0]
    if not features_with_na.empty:
        print(f"Filling NaN values in {len(features_with_na)} features with 0")
        for feature in features_with_na.index:
            train_df[feature] = train_df[feature].fillna(0)
    
    # Check data types and convert to float32 if needed
    float32_count = 0
    for feature in valid_features:
        if train_df[feature].dtype == np.float32:
            float32_count += 1
        else:
            try:
                train_df[feature] = train_df[feature].astype(np.float32)
            except Exception as e:
                print(f"Error converting {feature} to float32: {e}")
                train_df[feature] = pd.to_numeric(train_df[feature], errors='coerce').fillna(0).astype(np.float32)
    
    print(f"{float32_count}/{len(valid_features)} features were already float32")
    
    # Convert target to float32
    if train_df[main_target].dtype != np.float32:
        train_df[main_target] = train_df[main_target].astype(np.float32)
    
    # Also prepare validation data
    if val_df is not None:
        # Fill NaN values in target
        if val_df[main_target].isnull().sum() > 0:
            val_target_mean = val_df[main_target].mean()
            val_df[main_target] = val_df[main_target].fillna(val_target_mean)
        
        # Fill NaN values in features with 0
        val_valid_features = [f for f in valid_features if f in val_df.columns]
        for feature in val_valid_features:
            if val_df[feature].isnull().sum() > 0:
                val_df[feature] = val_df[feature].fillna(0)
            
            # Convert to float32
            if val_df[feature].dtype != np.float32:
                try:
                    val_df[feature] = val_df[feature].astype(np.float32)
                except:
                    val_df[feature] = pd.to_numeric(val_df[feature], errors='coerce').fillna(0).astype(np.float32)
        
        # Convert target to float32
        if val_df[main_target].dtype != np.float32:
            val_df[main_target] = val_df[main_target].astype(np.float32)

    # -------------------------------------------------
    # 1. Evaluate domain performance using correlation
    # -------------------------------------------------
    print("Evaluating domain performance with robust correlation calculation...")
    domain_scores = {}
    
    # Define a robust correlation function
    def robust_correlation(series1, series2):
        # Remove any remaining NaN values
        mask = ~np.isnan(series1) & ~np.isnan(series2)
        if mask.sum() < 10:  # Need at least 10 valid pairs for meaningful correlation
            return 0.0
        
        try:
            # Calculate correlation using numpy directly
            corr = np.corrcoef(series1[mask], series2[mask])[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except Exception as e:
            print(f"Error in correlation calculation: {e}")
            return 0.0
    
    for domain_name, feats in feature_groups.items():
        # Find valid features that exist in the data
        valid_domain_feats = [f for f in feats if f in train_df.columns]
        
        # Skip if too few features
        if len(valid_domain_feats) < 5:
            print(f"  {domain_name}: Too few valid features ({len(valid_domain_feats)}), skipping")
            domain_scores[domain_name] = 0.0
            continue
        
        # Calculate mean absolute correlation with target
        correlations = []
        
        for feat in valid_domain_feats:
            try:
                # Use our robust correlation function
                corr = robust_correlation(train_df[feat], train_df[main_target])
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception as e:
                print(f"  Error calculating correlation for {feat}: {e}")
        
        # Calculate mean correlation
        if correlations:
            mean_corr = np.mean(correlations)
            domain_scores[domain_name] = float(mean_corr)
            print(f"  {domain_name}: {mean_corr:.4f} (from {len(correlations)} valid features)")
        else:
            print(f"  {domain_name}: No valid correlations, setting score to 0")
            domain_scores[domain_name] = 0.0
    
    print("Domain performance scores (correlation with target):")
    for d, s in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
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
        print(f"Top domains selected: {kept_domains}")
        
    # Gather all features from kept domains
    kept_features = []
    for d in kept_domains:
        domain_feats = feature_groups[d]
        valid_feats = [f for f in domain_feats if f in train_df.columns]
        kept_features.extend(valid_feats)
    kept_features = list(set(kept_features))  # Remove duplicates
    
    # Ensure features exist in both train and val datasets
    if val_df is not None:
        valid_features = [f for f in kept_features if f in train_df.columns and f in val_df.columns]
        print(f"Total features from kept domains: {len(valid_features)}/{len(kept_features)} valid features")
        kept_features = valid_features
    else:
        valid_features = [f for f in kept_features if f in train_df.columns]
        print(f"Total features from kept domains: {len(valid_features)}/{len(kept_features)} valid features (validation not checked)")
        kept_features = valid_features

    # -------------------------------------------------
    # 3. Correlation pruning within kept features
    # -------------------------------------------------
    print(f"\nPruning correlated features (threshold={correlation_threshold})...")
    try:
        print("Using robust correlation pruning approach...")
        
        # Calculate correlation with target for all features
        print("Calculating feature-target correlations...")
        target_corrs = {}
        for feat in kept_features:
            target_corrs[feat] = robust_correlation(train_df[feat], train_df[main_target])
        
        # Sort features by correlation with target (descending)
        sorted_features = sorted(target_corrs.keys(), key=lambda x: target_corrs[x], reverse=True)
        
        # Greedy feature selection
        pruned_features = []
        remaining = sorted_features.copy()
        
        print(f"Pruning {len(remaining)} features...")
        while remaining:
            # Add the feature with highest target correlation
            if not remaining:
                break
                
            current = remaining.pop(0)
            pruned_features.append(current)
            
            # Find highly correlated features to remove
            to_remove = []
            for other in remaining:
                # Check if other feature is highly correlated with current
                corr = robust_correlation(train_df[current], train_df[other])
                if corr > correlation_threshold:
                    to_remove.append(other)
            
            # Remove highly correlated features
            remaining = [f for f in remaining if f not in to_remove]
        
        print(f"Pruning complete. Kept {len(pruned_features)} features out of {len(kept_features)}")
            
    except Exception as e:
        print(f"Error in correlation pruning: {e}")
        traceback.print_exc()
        print("Falling back to original features")
        pruned_features = kept_features

# Extract the prune_correlated_features function to reuse for domain-specific pruning
    def prune_correlated_features(df, features, target_col, corr_threshold):
        """Prune highly correlated features within a feature set"""
        # Calculate correlation with target
        target_correlations = {}
        for feat in features:
            target_correlations[feat] = robust_correlation(df[feat], df[target_col])
        
        # Sort by correlation with target
        sorted_feats = sorted(target_correlations.keys(), 
                              key=lambda x: target_correlations[x], 
                              reverse=True)
        
        # Greedy feature selection
        pruned = []
        remaining = sorted_feats.copy()
        
        while remaining:
            if not remaining:
                break
                
            # Add feature with highest target correlation
            current = remaining.pop(0)
            pruned.append(current)
            
            # Find highly correlated features to remove
            to_remove = []
            for other in remaining:
                corr = robust_correlation(df[current], df[other])
                if corr > corr_threshold:
                    to_remove.append(other)
            
            # Remove correlated features
            remaining = [f for f in remaining if f not in to_remove]
        
        return pruned

    # -------------------------------------------------
    # 4. Train domain-specific models
    # -------------------------------------------------
    print("\nTraining domain-specific models...")
    domain_models = {}
    domain_performances = {}
    ensemble_predictions = None
    ensemble_weights = {}
    
    for domain in kept_domains:
        try:
            # Get domain features that exist in the data
            domain_features = [f for f in feature_groups[domain] if f in train_df.columns]
            
            if len(domain_features) < 5:
                print(f"Domain {domain} has too few features ({len(domain_features)}), skipping model")
                continue
                
            # Prune correlated features within this domain
            print(f"Pruning features for domain {domain}...")
            domain_pruned = prune_correlated_features(
                train_df, domain_features, main_target, correlation_threshold
            )
            
            if len(domain_pruned) < 5:
                print(f"Domain {domain} has too few features after pruning ({len(domain_pruned)}), skipping model")
                continue
                
            print(f"Training model for domain {domain} with {len(domain_pruned)} features")
            
            # Train domain-specific model
            domain_model = LGBMRegressor(
                n_estimators=250,
                learning_rate=0.01,
                max_depth=5,
                num_leaves=25,
                colsample_bytree=0.7,
                subsample=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
            
            # Prepare data
            X_domain = train_df[domain_pruned].fillna(0).astype(np.float32)
            y_domain = train_df[main_target].fillna(0).astype(np.float32)
            
            # Train model
            domain_model.fit(X_domain, y_domain, eval_metric='rmse')
            
            # Evaluate model
            if val_df is not None:
                X_val_domain = val_df[domain_pruned].fillna(0).astype(np.float32)
                y_val_domain = val_df[main_target].fillna(0).astype(np.float32)
                
                domain_preds = domain_model.predict(X_val_domain)
                domain_score = robust_correlation(y_val_domain.values, domain_preds)
                
                # Calculate ensemble weight based on domain score
                weight = max(0, domain_score)  # Use validation score as weight
                
                # Store predictions for ensemble
                if ensemble_predictions is None:
                    ensemble_predictions = np.zeros_like(domain_preds)
                
                # Add weighted predictions to ensemble
                if weight > 0:
                    ensemble_predictions += weight * domain_preds
                    ensemble_weights[domain] = weight
            else:
                # Evaluate on training data if no validation set
                domain_preds = domain_model.predict(X_domain)
                domain_score = robust_correlation(y_domain.values, domain_preds)
                weight = max(0, domain_score)
            
            print(f"Domain {domain} model score: {domain_score:.4f}, weight: {weight:.4f}")
            
            # Calculate feature importance
            domain_importance = pd.DataFrame({
                'feature': domain_pruned,
                'importance': domain_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store domain model info
            domain_models[domain] = {
                'model': domain_model,
                'features': domain_pruned,
                'score': domain_score,
                'weight': weight,
                'importance': domain_importance
            }
            domain_performances[domain] = domain_score
            
            # Save model to disk
            try:
                import joblib
                import os
                
                # Create directories if they don't exist
                os.makedirs('domain_models', exist_ok=True)
                
                # Save model
                joblib.dump(domain_model, f'domain_models/{domain}_model.joblib')
                
                # Save features and metadata
                with open(f'domain_models/{domain}_metadata.json', 'w') as f:
                    json.dump({
                        'domain': domain,
                        'features': domain_pruned,
                        'score': float(domain_score),
                        'weight': float(weight),
                        'importance': [
                            {'feature': feat, 'importance': float(imp)}
                            for feat, imp in zip(domain_importance['feature'], domain_importance['importance'])
                        ]
                    }, f, indent=2)
                
                print(f"Saved model and metadata for domain {domain}")
                
            except Exception as save_error:
                print(f"Error saving domain model: {save_error}")
        
        except Exception as domain_error:
            print(f"Error training model for domain {domain}: {domain_error}")
    
    # -------------------------------------------------
    # 5. Create ensemble model
    # -------------------------------------------------
    ensemble_score = 0.0
    spearman_ensemble = 0.0
    
    if val_df is not None and ensemble_predictions is not None:
        # Normalize ensemble predictions
        total_weight = sum(ensemble_weights.values())
        if total_weight > 0:
            ensemble_predictions /= total_weight
            
            # Calculate ensemble performance
            ensemble_score = robust_correlation(val_df[main_target].values, ensemble_predictions)
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            spearman_ensemble = spearmanr(val_df[main_target].values, ensemble_predictions)[0]
            
            print(f"\nEnsemble model performance:")
            print(f"Ensemble correlation score: {ensemble_score:.4f}")
            print(f"Ensemble Spearman correlation: {spearman_ensemble:.4f}")
            
            # Save ensemble weights
            try:
                with open('domain_models/ensemble_weights.json', 'w') as f:
                    json.dump({
                        'domains': {k: float(v) for k, v in ensemble_weights.items()},
                        'total_weight': float(total_weight),
                        'correlation': float(ensemble_score),
                        'spearman': float(spearman_ensemble)
                    }, f, indent=2)
                print("Saved ensemble weights")
            except Exception as e:
                print(f"Error saving ensemble weights: {e}")

    # -------------------------------------------------
    # 6. Train a final global regression model on pruned features
    # -------------------------------------------------
    print("\nTraining final global regression model with pruned features...")
    try:
        # Check if we have enough pruned features
        if len(pruned_features) < 5:
            print(f"Warning: Too few pruned features ({len(pruned_features)}), adding more features")
            # Add more features to ensure we have at least 5
            additional_features = [f for f in kept_features if f not in pruned_features]
            additional_features = sorted(additional_features, 
                                        key=lambda f: target_corrs.get(f, 0),
                                        reverse=True)
            pruned_features.extend(additional_features[:max(5 - len(pruned_features), 0)])
            print(f"Added features to reach {len(pruned_features)} total features")
            
        if len(pruned_features) == 0:
            raise ValueError("No valid features remaining after pruning")
            
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
        
        # Prepare training data ensuring float32 type and no NaNs
        X_train = train_df[pruned_features].fillna(0).astype(np.float32)
        y_train = train_df[main_target].fillna(0).astype(np.float32)
        
        print(f"Training final model on {len(X_train)} samples with {len(pruned_features)} features...")
        
        # Check for any remaining issues
        if X_train.isna().any().any():
            print("Warning: X_train still has NaN values after fillna. Fixing...")
            X_train = X_train.fillna(0)
            
        if y_train.isna().any():
            print("Warning: y_train still has NaN values after fillna. Fixing...")
            y_train = y_train.fillna(y_train.mean())
        
        # Train model
        final_model.fit(
            X_train, 
            y_train,
            eval_metric='rmse'
        )
        
        # Save final model
        try:           
            os.makedirs('models', exist_ok=True)
            joblib.dump(final_model, 'models/final_model.joblib')
            
            # Save features and metadata
            with open('models/final_model_metadata.json', 'w') as f:
                json.dump({
                    'features': pruned_features,
                    'domains_used': kept_domains,
                    'score': float(final_score) if 'final_score' in locals() else 0.0,
                    'spearman': float(spearman_corr) if 'spearman_corr' in locals() else 0.0
                }, f, indent=2)
            
            print("Saved final model and metadata")
        except Exception as save_error:
            print(f"Error saving final model: {save_error}")
        
        # Generate predictions and calculate correlation
        if val_df is not None:
            # Prepare validation data
            X_val = val_df[pruned_features].fillna(0).astype(np.float32)
            y_val = val_df[main_target].fillna(0).astype(np.float32)
            
            # Make predictions
            val_preds = final_model.predict(X_val)
            
            # Calculate correlation
            final_score = robust_correlation(y_val.values, val_preds)
            print(f"Final model correlation score: {final_score:.4f}")
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            spearman_corr = spearmanr(y_val.values, val_preds)[0]
            print(f"Spearman correlation: {spearman_corr:.4f}")
            
            # Compare with ensemble
            if ensemble_score > 0:
                if ensemble_score > final_score:
                    print(f"Ensemble model outperforms global model: {ensemble_score:.4f} vs {final_score:.4f}")
                else:
                    print(f"Global model outperforms ensemble: {final_score:.4f} vs {ensemble_score:.4f}")
        else:
            # No validation data, use training data for evaluation
            print("No validation data provided, evaluating on training data")
            train_preds = final_model.predict(X_train)
            final_score = robust_correlation(y_train.values, train_preds)
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            spearman_corr = spearmanr(y_train.values, train_preds)[0]
            
            print(f"Final model correlation score (on training): {final_score:.4f}")
            print(f"Spearman correlation (on training): {spearman_corr:.4f}")
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': pruned_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        print(f"Error in final model training: {e}")
        traceback.print_exc()
        print("Returning partial results without final model")
        final_score = 0.0
        spearman_corr = 0.0
        feature_importance = pd.DataFrame({'feature': pruned_features, 'importance': 0.0})

    # -------------------------------------------------
    # 7. Return results
    # -------------------------------------------------
    results = {
        'domain_scores': domain_scores,
        'kept_domains': kept_domains,
        'pruned_features': pruned_features,
        'final_model_score': final_score,
        'spearman_correlation': spearman_corr,
        'feature_importance': feature_importance.head(20) if len(feature_importance) > 0 else None,
        'domain_models': {
            domain: {
                'model': info['model'],
                'features': info['features'],
                'score': info['score'],
                'weight': info['weight']
            } for domain, info in domain_models.items()
        },
        'ensemble_score': ensemble_score,
        'ensemble_spearman': spearman_ensemble,
        'ensemble_weights': ensemble_weights
    }
    return results

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
    random_seed: int = 42
) -> Dict:
    """
    Enhanced domain pipeline with proper float32 conversion and memory optimization.
    
    Args:
        data_version: Version of dataset
        feature_set: Size of feature set (small, medium, large)
        main_target: Primary target column
        num_aux_targets: Number of auxiliary targets to use
        aux_targets: Specific auxiliary targets to use (overrides num_aux_targets)
        sample_size: Number of samples to use (None for all)
        n_clusters: Number of clusters (None for automatic determination)
        save_path: Path to save domain data
        chunk_size: Size of chunks for processing large datasets
        use_incremental: Whether to use incremental processing for large datasets
        skip_visualizations: Whether to skip generating visualizations
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict containing domain information
    """
    print(f"Starting enhanced domain pipeline with {feature_set} feature set")
    log_memory_usage("Initial")
    results = {}

    try:
        # ---------------------------
        # 1. Data Loading & Preparation with float32 conversion
        # ---------------------------
        print("Loading data with explicit float32 conversion...")
        
        # Set memory optimization flags based on dataset size
        if feature_set == "large":
            use_incremental = True
            print("Large feature set detected, enabling incremental processing")
        
        if aux_targets is not None:
            print(f"Using specified auxiliary targets: {aux_targets}")
            train_df, val_df, features, all_targets = load_data(
                data_version=data_version,
                feature_set=feature_set,
                main_target=main_target,
                aux_targets=aux_targets,  # Pass the aux_targets directly
                num_aux_targets=0  # Set to 0 to prevent random selection
            )
            # Check if all specified targets are in the loaded data
            available_targets = [col for col in train_df.columns if col.startswith('target_')]
            print(f"Available targets: {available_targets}")
            missing_targets = [t for t in aux_targets if t not in train_df.columns]
            if missing_targets:
                raise ValueError(f"Specified auxiliary targets not found in data: {missing_targets}")
            all_targets = [main_target] + aux_targets  # Use the specified targets
        else:
            print(f"Selecting {num_aux_targets} random auxiliary targets")
            train_df, val_df, features, all_targets = load_data(
                data_version=data_version,
                feature_set=feature_set,
                main_target=main_target,
                num_aux_targets=num_aux_targets
            )
        
        if train_df is None or val_df is None:
            raise ValueError("Failed to load data")
            
        print(f"Loaded data with {len(features)} features")
        print(f"Using targets: {all_targets}")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
        
        # Explicitly convert features and targets to float32
        print("Converting features and targets to float32...")
        for feature in features:
            if feature in train_df.columns:
                train_df[feature] = train_df[feature].astype(np.float32)
            if feature in val_df.columns:
                val_df[feature] = val_df[feature].astype(np.float32)
                
        for target in all_targets:
            if target in train_df.columns:
                train_df[target] = train_df[target].astype(np.float32)
            if target in val_df.columns:
                val_df[target] = val_df[target].astype(np.float32)
        
        log_memory_usage("After data type conversion")

        # Apply sampling if specified
        if sample_size is not None and sample_size < len(train_df):
            print(f"Sampling {sample_size} rows from training data")
            train_df = train_df.sample(n=sample_size, random_state=random_seed)
        
        # ---------------------------
        # 2. Memory-Efficient Correlation Computation
        # ---------------------------
        print("Computing feature-target correlations with memory optimization...")
        
        if use_incremental and len(features) * len(all_targets) > 10000:
            print("Using incremental correlation computation for large feature set")
            # Process in batches to avoid memory issues
            batch_size = min(500, len(features))
            correlations = []
            
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_corrs = []
                
                for feature in batch_features:
                    feature_corrs = []
                    # Handle NaNs by filling with 0
                    feature_data = train_df[feature].fillna(0).values.astype(np.float32)
                    
                    for target in all_targets:
                        target_data = train_df[target].fillna(0).values.astype(np.float32)
                        # Use memory-efficient correlation calculation
                        corr = np.corrcoef(feature_data, target_data)[0, 1]
                        feature_corrs.append(corr)
                    
                    batch_corrs.append(feature_corrs)
                
                correlations.extend(batch_corrs)
                
                # Free memory after each batch
                gc.collect()
                log_memory_usage(f"After batch {i//batch_size + 1}/{(len(features) + batch_size - 1)//batch_size}")
        else:
            # Standard processing for smaller datasets
            correlations = []
            for feature in features:
                feature_corrs = []
                for target in all_targets:
                    # Handle NaNs by filling with 0
                    feature_data = train_df[feature].fillna(0).values.astype(np.float32)
                    target_data = train_df[target].fillna(0).values.astype(np.float32)
                    corr = np.corrcoef(feature_data, target_data)[0, 1]
                    feature_corrs.append(corr)
                correlations.append(feature_corrs)
        
        correlations = np.array(correlations, dtype=np.float32)
        print(f"Correlation matrix shape: {correlations.shape}")
        
        # ---------------------------
        # 3. Enhanced Embeddings using t-SNE with memory optimization
        # ---------------------------
        print("Applying t-SNE for enhanced feature embeddings...")
        # Free memory before t-SNE
        gc.collect()
        log_memory_usage("Before t-SNE")
        
        # Use early_exaggeration=4.0 for memory optimization
        tsne = TSNE(
            n_components=3, 
            perplexity=min(30, len(features) - 1),  # Adjust perplexity for small feature sets
            random_state=random_seed,
            n_jobs=-1,  # Use all available cores
            early_exaggeration=4.0,  # Lower value for memory optimization
            method='barnes_hut' if len(features) > 1000 else 'exact'  # Memory-efficient for large sets
        )
        feature_embeddings = tsne.fit_transform(correlations)
        feature_embeddings = feature_embeddings.astype(np.float32)  # Ensure float32
        print(f"Feature embeddings shape: {feature_embeddings.shape}")
        
        # Free memory after t-SNE
        del correlations
        gc.collect()
        log_memory_usage("After t-SNE")

        # ---------------------------
        # 4. Dynamic Cluster Determination with memory optimization
        # ---------------------------
        if n_clusters is None:
            print("Determining optimal number of clusters dynamically using silhouette scores...")
            silhouette_scores = []
            
            # Adjust k range based on feature set size
            if len(features) < 100:
                k_range = range(max(3, len(features) // 10), min(15, len(features) // 2))
            elif len(features) < 500:
                k_range = range(10, 20)
            else:
                k_range = range(15, 30)
                
            for k in k_range:
                # Use batch_size=None for small datasets, otherwise use memory-efficient MiniBatchKMeans
                if len(features) < 1000:
                    clusterer = KMeans(n_clusters=k, random_state=random_seed, n_init=5)
                else:
                    from sklearn.cluster import MiniBatchKMeans
                    clusterer = MiniBatchKMeans(
                        n_clusters=k, 
                        random_state=random_seed,
                        batch_size=min(1000, len(features)),
                        n_init=3
                    )
                    
                labels = clusterer.fit_predict(feature_embeddings)
                score = silhouette_score(feature_embeddings, labels)
                silhouette_scores.append(score)
                print(f"Silhouette score for k={k}: {score:.4f}")
                
                # Free memory after each iteration
                gc.collect()
                
            optimal_k = k_range[np.argmax(silhouette_scores)]
            n_clusters = optimal_k
            print(f"Optimal number of clusters determined: {n_clusters}")
        else:
            print(f"Using user-specified number of clusters: {n_clusters}")

        # ---------------------------
        # 5. Memory-Efficient Clustering
        # ---------------------------
        print(f"Clustering features into {n_clusters} domains...")
        
        # Use appropriate clustering algorithm based on dataset size
        if len(features) < 1000:
            clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=random_seed,
                n_init=10
            )
        else:
            from sklearn.cluster import MiniBatchKMeans
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=random_seed,
                batch_size=min(1000, len(features)),
                n_init=3
            )
            
        cluster_labels = clusterer.fit_predict(feature_embeddings)

        # Create feature groups (domains)
        feature_groups = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            domain_name = f"domain_{label}"
            feature_groups[domain_name].append(features[idx])
        feature_groups = dict(feature_groups)
        
        # Print domain sizes
        print("Domain sizes:")
        for domain, domain_features in feature_groups.items():
            print(f"  {domain}: {len(domain_features)} features")

        # ---------------------------
        # 6. Save Domain Data
        # ---------------------------
        results['feature_groups'] = feature_groups
        results['num_domains'] = len(feature_groups)
        results['targets_used'] = all_targets
        results['embeddings_shape'] = feature_embeddings.shape

        try:
            print("Saving domain data...")
            # Create domain data in batches if there are many features
            if len(features) > 10000:
                print("Creating domain data in batches...")
                domains_df_list = []
                batch_size = 5000
                
                for i in range(0, len(features), batch_size):
                    batch_domain_data = []
                    batch_end = min(i + batch_size, len(features))
                    
                    for idx in range(i, batch_end):
                        batch_domain_data.append({
                            'feature': features[idx],
                            'domain_id': int(cluster_labels[idx]),
                            'domain_name': f"domain_{int(cluster_labels[idx])}",
                            **{f'dimension_{i+1}': emb.item() for i, emb in enumerate(feature_embeddings[idx])}
                        })
                    
                    domains_df_list.append(pd.DataFrame(batch_domain_data))
                    
                # Concatenate all batches
                domains_df = pd.concat(domains_df_list, ignore_index=True)
            else:
                # Standard processing for smaller datasets
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
            
            # Save a summary file with domain statistics
            summary_path = save_path.replace('.csv', '_summary.csv')
            domain_summary = []
            for domain_id in range(n_clusters):
                domain_features = [features[i] for i, label in enumerate(cluster_labels) if label == domain_id]
                domain_summary.append({
                    'domain_id': domain_id,
                    'domain_name': f"domain_{domain_id}",
                    'feature_count': len(domain_features),
                    'sample_features': ", ".join(domain_features[:5]) + ("..." if len(domain_features) > 5 else "")
                })
            pd.DataFrame(domain_summary).to_csv(summary_path, index=False)
            results['domain_summary_path'] = summary_path
            
        except Exception as save_error:
            print(f"Error saving domain data: {save_error}")
            results['save_error'] = str(save_error)
        
        # ---------------------------
        # 7. Memory-Efficient Cluster Validation
        # ---------------------------
        print("Computing intra-domain correlation matrix...")
        
        # Use memory-efficient approach for large datasets
        if len(features) > 5000 or len(train_df) > 50000:
            print("Using memory-efficient domain validation approach")
            
            # Process one domain at a time
            domain_corr_matrix = np.zeros((n_clusters, len(all_targets)), dtype=np.float32)
            
            for domain_id in range(n_clusters):
                domain_features = [features[i] for i, label in enumerate(cluster_labels) if label == domain_id]
                
                if len(domain_features) == 0:
                    continue
                    
                # Process one target at a time
                for j, target in enumerate(all_targets):
                    # Calculate correlations in mini-batches
                    batch_size = min(200, len(domain_features))
                    domain_corrs = []
                    
                    for k in range(0, len(domain_features), batch_size):
                        batch_features = domain_features[k:k+batch_size]
                        batch_corrs = []
                        
                        for feat in batch_features:
                            if feat in train_df.columns and target in train_df.columns:
                                feat_data = train_df[feat].fillna(0).values.astype(np.float32)
                                target_data = train_df[target].fillna(0).values.astype(np.float32)
                                corr = np.corrcoef(feat_data, target_data)[0, 1]
                                batch_corrs.append(corr)
                        
                        domain_corrs.extend(batch_corrs)
                        
                    # Calculate mean correlation for this domain-target pair
                    if domain_corrs:
                        domain_corr_matrix[domain_id, j] = np.mean(domain_corrs)
                    else:
                        domain_corr_matrix[domain_id, j] = 0
                        
                # Free memory after processing each domain
                gc.collect()
        else:
            # Standard approach for smaller datasets
            domain_corr_matrix = np.zeros((n_clusters, len(all_targets)))
            for domain_id in range(n_clusters):
                domain_features = [features[i] for i, label in enumerate(cluster_labels) if label == domain_id]
                for j, target in enumerate(all_targets):
                    corrs = []
                    for feat in domain_features:
                        if feat in train_df.columns and target in train_df.columns:
                            feat_data = train_df[feat].fillna(0).values.astype(np.float32)
                            target_data = train_df[target].fillna(0).values.astype(np.float32)
                            corr = np.corrcoef(feat_data, target_data)[0, 1]
                            corrs.append(corr)
                    if corrs:
                        domain_corr_matrix[domain_id, j] = np.mean(corrs)
        
        # Save validation results
        results['domain_validation'] = domain_corr_matrix.tolist()
        
        # Free memory before visualization
        gc.collect()
        log_memory_usage("Before visualization")
        
        # ---------------------------
        # 8. Memory-Efficient Visualizations
        # ---------------------------
        if not skip_visualizations:
            try:
                print("Generating memory-efficient visualizations...")
                
                # Select a subset of features for visualization if dataset is very large
                if len(features) > 5000:
                    print(f"Subsampling {min(5000, len(features))} features for visualization")
                    indices = np.random.choice(len(features), size=min(5000, len(features)), replace=False)
                    vis_embeddings = feature_embeddings[indices]
                    vis_labels = cluster_labels[indices]
                else:
                    vis_embeddings = feature_embeddings
                    vis_labels = cluster_labels
                
                # Use UMAP with memory-efficient settings
                if len(vis_embeddings) > 1000:
                    print("Using memory-optimized UMAP settings")
                    reducer = umap.UMAP(
                        n_neighbors=min(15, len(vis_embeddings) // 10),
                        min_dist=0.1,
                        metric='euclidean',
                        random_state=random_seed,
                        n_jobs=-1  # Use all available cores
                    )
                else:
                    reducer = umap.UMAP(random_state=random_seed)
                
                # Convert to float32 for memory efficiency
                vis_embeddings = vis_embeddings.astype(np.float32)
                
                # Generate 2D embeddings
                umap_embeddings = reducer.fit_transform(vis_embeddings)
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    umap_embeddings[:, 0], 
                    umap_embeddings[:, 1],
                    c=vis_labels, 
                    cmap='tab20', 
                    s=10,
                    alpha=0.7
                )
                plt.colorbar(scatter)
                plt.title(f"Feature Domains Visualization - {len(feature_groups)} Clusters")
                
                # Add legend with domain sizes
                domain_sizes = {}
                for domain_id in range(n_clusters):
                    count = np.sum(cluster_labels == domain_id)
                    domain_sizes[f"Domain {domain_id}"] = count
                
                # Sort by size and add top 10 domains to legend
                top_domains = dict(sorted(domain_sizes.items(), key=lambda x: x[1], reverse=True)[:10])
                plt.figtext(0.02, 0.02, "Top domains by size:", fontsize=8)
                y_pos = 0.05
                for domain, size in top_domains.items():
                    plt.figtext(0.02, y_pos, f"{domain}: {size} features", fontsize=7)
                    y_pos += 0.02
                
                # Save the plot
                plot_path = save_path.replace('.csv', '_plot.png')
                plt.savefig(plot_path, dpi=120, bbox_inches='tight')
                plt.close()
                results['plot_path'] = plot_path
                                
                print("Visualizations generated successfully")
            except Exception as viz_error:
                print(f"Error generating visualizations: {viz_error}")
                results['visualization_error'] = str(viz_error)
        
        # Final memory cleanup
        gc.collect()
        log_memory_usage("Final")
        
        return results

    except Exception as e:
        print(f"Pipeline error: {e}")
        traceback.print_exc()
        return {'error': str(e)}