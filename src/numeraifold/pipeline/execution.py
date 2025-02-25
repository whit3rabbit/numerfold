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

    Returns:
        dict: Contains domain_scores, kept_domains, final_model_score, pruned_features, etc.
    """

    # -------------------------------------------------
    # 1. Evaluate domain performance using correlation
    # -------------------------------------------------
    print("Evaluating domain performance...")
    domain_scores = {}
    for domain_name, feats in feature_groups.items():
        # Handle empty or tiny feature groups
        if len(feats) < 5:
            print(f"  {domain_name}: Too few features ({len(feats)}), skipping")
            domain_scores[domain_name] = 0.0
            continue
            
        # Calculate mean absolute correlation with target
        try:
            correlations = train_df[feats].corrwith(train_df[main_target]).abs()
            mean_corr = correlations.mean()
            domain_scores[domain_name] = float(mean_corr)
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
    try:
        # Fit the selector on training data's kept features
        selector = SmartCorrelatedSelection(
            threshold=correlation_threshold, 
            variables=kept_features,
            method='pearson'
        )
        # Only pass the required columns to avoid errors
        columns_to_use = kept_features + [main_target]
        train_subset = train_df[columns_to_use].copy()
        train_pruned = selector.fit_transform(train_subset)
        pruned_features = [f for f in train_pruned.columns if f != main_target]
        
        print(f"Reduced from {len(kept_features)} to {len(pruned_features)} features after pruning")
    except Exception as e:
        print(f"Error in correlation pruning: {e}")
        print("Falling back to original features")
        pruned_features = kept_features

    # -------------------------------------------------
    # 4. Train a final global regression model on pruned features
    # -------------------------------------------------
    print("\nTraining final global regression model with pruned features...")
    try:
        # Check if pruned features exist in both datasets
        valid_features = [f for f in pruned_features if f in train_df.columns and f in val_df.columns]
        if len(valid_features) < len(pruned_features):
            print(f"Warning: {len(pruned_features) - len(valid_features)} features not found in both datasets")
            pruned_features = valid_features
            
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
        
        final_model.fit(
            train_df[pruned_features], 
            train_df[main_target],
            eval_metric='rmse'
        )
        
        # Generate predictions and calculate correlation
        val_preds = final_model.predict(val_df[pruned_features])
        final_score = np.corrcoef(val_df[main_target].values, val_preds)[0, 1]
        print(f"Final model correlation score: {final_score:.4f}")
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': pruned_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        print(f"Error in final model training: {e}")
        print("Returning partial results without final model")
        final_score = 0.0
        feature_importance = pd.DataFrame({'feature': pruned_features, 'importance': 0.0})

    # -------------------------------------------------
    # 5. Return results
    # -------------------------------------------------
    results = {
        'domain_scores': domain_scores,
        'kept_domains': kept_domains,
        'pruned_features': pruned_features,
        'final_model_score': final_score,
        'feature_importance': feature_importance.head(20) if len(feature_importance) > 0 else None
    }
    return results

def follow_up_domains_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: dict,
    main_target: str,
    domain_score_threshold: float = 0.01,  # Lower threshold for correlation
    correlation_threshold: float = 0.95
):
    """
    Follow-up pipeline that ensures proper float32 data type conversion:
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
    print("Starting follow-up domains pipeline with float32 data type enforcement...")
    
    # -------------------------------------------------
    # 0. Ensure data types are float32
    # -------------------------------------------------
    print("Converting features to float32...")
    all_features = []
    for feats in feature_groups.values():
        all_features.extend(feats)
    all_features = list(set(all_features))  # Remove duplicates
    
    # Convert features to float32 in both datasets
    for feature in all_features:
        if feature in train_df.columns:
            train_df[feature] = train_df[feature].astype(np.float32)
        if feature in val_df.columns:
            val_df[feature] = val_df[feature].astype(np.float32)
    
    # Convert target to float32
    if main_target in train_df.columns:
        train_df[main_target] = train_df[main_target].astype(np.float32)
    if main_target in val_df.columns:
        val_df[main_target] = val_df[main_target].astype(np.float32)
    
    memory_usage_train = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_usage_val = val_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage - Train: {memory_usage_train:.2f} MB, Val: {memory_usage_val:.2f} MB")

    # -------------------------------------------------
    # 1. Evaluate domain performance using correlation
    # -------------------------------------------------
    print("Evaluating domain performance...")
    domain_scores = {}
    for domain_name, feats in feature_groups.items():
        # Handle empty or tiny feature groups
        if len(feats) < 5:
            print(f"  {domain_name}: Too few features ({len(feats)}), skipping")
            domain_scores[domain_name] = 0.0
            continue
            
        # Calculate mean absolute correlation with target
        try:
            # Ensure features exist in train_df
            valid_feats = [f for f in feats if f in train_df.columns]
            if len(valid_feats) < 5:
                print(f"  {domain_name}: Too few valid features ({len(valid_feats)}), skipping")
                domain_scores[domain_name] = 0.0
                continue
                
            # Calculate in chunks to avoid memory issues with large datasets
            if len(train_df) > 50000:
                print(f"  Large dataset detected, calculating correlation in chunks for {domain_name}...")
                correlations = []
                chunk_size = 10000
                for i in range(0, len(train_df), chunk_size):
                    chunk = train_df.iloc[i:i+chunk_size]
                    chunk_corr = chunk[valid_feats].corrwith(chunk[main_target]).abs()
                    correlations.append(chunk_corr)
                # Average correlations across chunks
                mean_corr = pd.concat(correlations, axis=1).mean(axis=1).mean()
            else:
                correlations = train_df[valid_feats].corrwith(train_df[main_target]).abs()
                mean_corr = correlations.mean()
                
            domain_scores[domain_name] = float(mean_corr)
            print(f"  {domain_name}: {mean_corr:.4f} (from {len(valid_feats)} features)")
        except Exception as e:
            print(f"  Error evaluating {domain_name}: {e}")
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
        
    # Gather all features from kept domains
    kept_features = []
    for d in kept_domains:
        kept_features.extend(feature_groups[d])
    kept_features = list(set(kept_features))  # Remove duplicates
    
    # Ensure features exist in both train and val datasets
    valid_features = [f for f in kept_features if f in train_df.columns and f in val_df.columns]
    print(f"Total features from kept domains: {len(valid_features)}/{len(kept_features)} valid features")
    kept_features = valid_features

    # -------------------------------------------------
    # 3. Correlation pruning within kept features
    # -------------------------------------------------
    print(f"\nPruning correlated features (threshold={correlation_threshold})...")
    try:
        # Handle large datasets by using a correlation-based feature selection approach
        if len(train_df) > 50000 or len(kept_features) > 500:
            print("Large dataset detected, using memory-efficient correlation pruning...")
            # Use a more memory-efficient approach for large datasets
            pruned_features = []
            remaining_features = kept_features.copy()
            
            # Calculate correlation with target for sorting
            target_corrs = {}
            for f in remaining_features:
                # Calculate in smaller chunks if needed
                if len(train_df) > 50000:
                    corrs = []
                    chunk_size = 10000
                    for i in range(0, len(train_df), chunk_size):
                        chunk = train_df.iloc[i:i+chunk_size]
                        chunk_corr = abs(np.corrcoef(chunk[f].fillna(0).astype(np.float32), 
                                                   chunk[main_target].fillna(0).astype(np.float32))[0, 1])
                        corrs.append(chunk_corr)
                    target_corrs[f] = np.mean(corrs)
                else:
                    target_corrs[f] = abs(np.corrcoef(train_df[f].fillna(0).astype(np.float32), 
                                                    train_df[main_target].fillna(0).astype(np.float32))[0, 1])
            
            # Sort features by correlation with target (descending)
            sorted_features = sorted(target_corrs.keys(), key=lambda x: target_corrs[x], reverse=True)
            
            # Greedy feature selection
            while sorted_features:
                # Add the most correlated feature to the pruned list
                current_feature = sorted_features.pop(0)
                pruned_features.append(current_feature)
                
                # Remove highly correlated features
                features_to_remove = []
                for other_feature in sorted_features:
                    # Calculate correlation in chunks if dataset is large
                    if len(train_df) > 50000:
                        corrs = []
                        chunk_size = 10000
                        for i in range(0, len(train_df), chunk_size):
                            chunk = train_df.iloc[i:i+chunk_size]
                            chunk_corr = abs(np.corrcoef(chunk[current_feature].fillna(0).astype(np.float32), 
                                                       chunk[other_feature].fillna(0).astype(np.float32))[0, 1])
                            corrs.append(chunk_corr)
                        corr = np.mean(corrs)
                    else:
                        corr = abs(np.corrcoef(train_df[current_feature].fillna(0).astype(np.float32), 
                                             train_df[other_feature].fillna(0).astype(np.float32))[0, 1])
                    
                    if corr > correlation_threshold:
                        features_to_remove.append(other_feature)
                
                # Remove features and update sorted list
                for f in features_to_remove:
                    if f in sorted_features:
                        sorted_features.remove(f)
        else:
            # Use the original SmartCorrelatedSelection for smaller datasets
            # Fit the selector on training data's kept features
            selector = SmartCorrelatedSelection(
                threshold=correlation_threshold, 
                variables=kept_features,
                method='pearson'
            )
            # Only pass the required columns to avoid errors
            columns_to_use = kept_features + [main_target]
            train_subset = train_df[columns_to_use].copy()
            
            # Apply selection
            selector.fit(train_subset)
            
            # Get features to drop and keep
            features_to_drop = selector.features_to_drop_
            pruned_features = [f for f in kept_features if f not in features_to_drop]
        
        print(f"Reduced from {len(kept_features)} to {len(pruned_features)} features after pruning")
    except Exception as e:
        print(f"Error in correlation pruning: {e}")
        print("Falling back to original features")
        pruned_features = kept_features

    # -------------------------------------------------
    # 4. Train a final global regression model on pruned features
    # -------------------------------------------------
    print("\nTraining final global regression model with pruned features...")
    try:
        # Check if pruned features exist in both datasets
        valid_features = [f for f in pruned_features if f in train_df.columns and f in val_df.columns]
        if len(valid_features) < len(pruned_features):
            print(f"Warning: {len(pruned_features) - len(valid_features)} features not found in both datasets")
            pruned_features = valid_features
            
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
        
        # Prepare training data ensuring float32 type
        X_train = train_df[pruned_features].fillna(0).astype(np.float32)
        y_train = train_df[main_target].fillna(0).astype(np.float32)
        
        # Train model
        final_model.fit(
            X_train, 
            y_train,
            eval_metric='rmse'
        )
        
        # Generate predictions and calculate correlation
        X_val = val_df[pruned_features].fillna(0).astype(np.float32)
        y_val = val_df[main_target].fillna(0).astype(np.float32)
        
        val_preds = final_model.predict(X_val)
        final_score = np.corrcoef(y_val.values, val_preds)[0, 1]
        print(f"Final model correlation score: {final_score:.4f}")
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': pruned_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        print(f"Error in final model training: {e}")
        print("Returning partial results without final model")
        final_score = 0.0
        feature_importance = pd.DataFrame({'feature': pruned_features, 'importance': 0.0})

    # -------------------------------------------------
    # 5. Return results
    # -------------------------------------------------
    results = {
        'domain_scores': domain_scores,
        'kept_domains': kept_domains,
        'pruned_features': pruned_features,
        'final_model_score': final_score,
        'feature_importance': feature_importance.head(20) if len(feature_importance) > 0 else None
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
