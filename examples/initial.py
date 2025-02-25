import gc
import time
import json
import os
import numpy as np
import pandas as pd
import psutil
import traceback
import torch
import joblib

from numeraifold.data.loading import load_data
from numeraifold.domains.models import make_predictions_pipeline
from numeraifold.pipeline.execution import run_domains_only_pipeline, follow_up_domains_pipeline
from numeraifold.utils.seed import set_seed

# Optional: Import numerai-tools for scoring if available
try:
    from numerapi import NumerAPI
    from numerai_tools.scoring import correlation_contribution
    NUMERAI_TOOLS_AVAILABLE = True
except ImportError:
    NUMERAI_TOOLS_AVAILABLE = False
    print("numerai-tools not available. Will use basic scoring methods.")


# Set seed and check CUDA
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
print(f"CUDA available: {torch.cuda.is_available()}")

def log_memory_usage(label="Current memory usage"):
    """Log the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    # Get GPU memory usage if available
    gpu_memory_usage = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_memory_usage = f"Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB"
        except:
            pass
    
    print(f"{label} - RAM: {memory_usage_mb:.2f} MB, GPU: {gpu_memory_usage}")

def load_domain_models(models_dir='domain_models'):
    """
    Load all saved domain models and their metadata
    
    Args:
        models_dir: Directory where domain models are saved
        
    Returns:
        dict: Dictionary of domain models and their metadata
    """
    if not os.path.exists(models_dir):
        print(f"Error: Models directory {models_dir} does not exist")
        return {}
    
    domain_models = {}
    
    # Load ensemble weights if available
    ensemble_weights = {}
    ensemble_weights_path = os.path.join(models_dir, 'ensemble_weights.json')
    if os.path.exists(ensemble_weights_path):
        try:
            with open(ensemble_weights_path, 'r') as f:
                ensemble_data = json.load(f)
                ensemble_weights = ensemble_data.get('domains', {})
                print(f"Loaded ensemble weights for {len(ensemble_weights)} domains")
        except Exception as e:
            print(f"Error loading ensemble weights: {e}")
    
    # Find all domain model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
    
    for model_file in model_files:
        try:
            # Extract domain name from filename
            domain = model_file.replace('_model.joblib', '')
            
            # Load model
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = os.path.join(models_dir, f"{domain}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                print(f"Warning: No metadata found for {domain}, using defaults")
                metadata = {
                    'domain': domain,
                    'features': [],
                    'score': 0.0,
                    'weight': ensemble_weights.get(domain, 0.0)
                }
            
            # Add model to dictionary
            domain_models[domain] = {
                'model': model,
                'features': metadata['features'],
                'score': metadata['score'],
                'weight': metadata.get('weight', ensemble_weights.get(domain, 0.0))
            }
            print(f"Loaded model for {domain} with {len(metadata['features'])} features")
            
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
    
    print(f"Successfully loaded {len(domain_models)} domain models")
    return domain_models

def predict_with_domain_models(df, domain_models, target_col=None):
    """
    Make predictions using domain models
    
    Args:
        df: DataFrame containing features
        domain_models: Dictionary of domain models from load_domain_models()
        target_col: Optional target column for evaluation
        
    Returns:
        dict: Dictionary containing predictions and evaluation metrics
    """
    # Initialize results
    results = {
        'domain_predictions': {},
        'ensemble_prediction': None,
        'domain_scores': {},
        'best_domain': None,
        'best_score': 0.0
    }
    
    if len(domain_models) == 0:
        print("No domain models available for prediction")
        return results
    
    # For ensemble prediction
    ensemble_pred = None
    total_weight = 0.0
    
    # Make predictions with each domain model
    for domain, model_info in domain_models.items():
        model = model_info['model']
        features = model_info['features']
        weight = model_info['weight']
        
        # Check if all features are available
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing for {domain}")
            if len(missing_features) / len(features) > 0.1:  # If more than 10% missing
                print(f"Skipping {domain} due to too many missing features")
                continue
        
        # Prepare features
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) < 5:
            print(f"Not enough valid features for {domain}, skipping")
            continue
            
        X = df[valid_features].fillna(0).astype(np.float32)
        
        # Make predictions
        try:
            domain_pred = model.predict(X)
            results['domain_predictions'][domain] = domain_pred
            
            # Add to ensemble if weight > 0
            if weight > 0:
                if ensemble_pred is None:
                    ensemble_pred = np.zeros_like(domain_pred)
                ensemble_pred += weight * domain_pred
                total_weight += weight
            
            # Evaluate if target column provided
            if target_col is not None and target_col in df.columns:
                y_true = df[target_col].fillna(0).astype(np.float32).values
                
                # Calculate correlation
                mask = ~np.isnan(domain_pred) & ~np.isnan(y_true)
                if mask.sum() >= 10:  # Need at least 10 valid pairs
                    corr = np.corrcoef(domain_pred[mask], y_true[mask])[0, 1]
                    results['domain_scores'][domain] = float(corr)
                    
                    # Track best domain
                    if corr > results['best_score']:
                        results['best_domain'] = domain
                        results['best_score'] = float(corr)
                        
        except Exception as e:
            print(f"Error making predictions with {domain}: {e}")
    
    # Normalize ensemble prediction
    if ensemble_pred is not None and total_weight > 0:
        ensemble_pred = ensemble_pred / total_weight
        results['ensemble_prediction'] = ensemble_pred
        
        # Evaluate ensemble if target provided
        if target_col is not None and target_col in df.columns:
            y_true = df[target_col].fillna(0).astype(np.float32).values
            
            # Calculate correlation
            mask = ~np.isnan(ensemble_pred) & ~np.isnan(y_true)
            if mask.sum() >= 10:
                corr = np.corrcoef(ensemble_pred[mask], y_true[mask])[0, 1]
                results['ensemble_score'] = float(corr)
                
                # Check if ensemble is best
                if corr > results['best_score']:
                    results['best_model'] = 'ensemble'
                    results['best_score'] = float(corr)
    
    # Print summary
    print(f"Made predictions with {len(results['domain_predictions'])} domain models")
    if 'ensemble_prediction' in results and results['ensemble_prediction'] is not None:
        print(f"Ensemble prediction created with {total_weight:.4f} total weight")
    if 'domain_scores' in results:
        print(f"Domain performance:")
        for domain, score in sorted(results['domain_scores'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {domain}: {score:.4f}")
        if 'ensemble_score' in results:
            print(f"Ensemble score: {results['ensemble_score']:.4f}")
    
    return results

def make_predictions_pipeline(new_data, target_col=None, models_dir='domain_models'):
    """
    End-to-end pipeline for making predictions on new data
    
    Args:
        new_data: DataFrame with features
        target_col: Optional target column for evaluation
        models_dir: Directory containing saved domain models
        
    Returns:
        DataFrame with predictions
    """
    # Load domain models
    domain_models = load_domain_models(models_dir)
    if not domain_models:
        print("No domain models found, cannot make predictions")
        return None
    
    # Make predictions
    results = predict_with_domain_models(new_data, domain_models, target_col)
    
    # Create output DataFrame - keep the original index
    output_df = pd.DataFrame(index=new_data.index)
    
    # Add individual domain predictions
    for domain, preds in results['domain_predictions'].items():
        output_df[f'pred_{domain}'] = preds
    
    # Add ensemble prediction
    if results['ensemble_prediction'] is not None:
        output_df['prediction_ensemble'] = results['ensemble_prediction']
    
    # Add best prediction
    if 'best_model' in results:
        if results['best_model'] == 'ensemble':
            output_df['prediction_best'] = results['ensemble_prediction']
        else:
            output_df['prediction_best'] = results['domain_predictions'][results['best_domain']]
    elif 'best_domain' in results and results['best_domain'] is not None:
        output_df['prediction_best'] = results['domain_predictions'][results['best_domain']]
    
    # Copy important columns from the input data
    # This is key - we need to preserve era and other metadata columns
    important_cols = ['era'] if 'era' in new_data.columns else []
    if target_col is not None and target_col in new_data.columns:
        important_cols.append(target_col)
    
    # Add any ID columns
    if 'id' in new_data.columns:
        important_cols.append('id')
    
    # Copy these columns to the output DataFrame
    for col in important_cols:
        output_df[col] = new_data[col]
    
    # Log available columns for debugging
    print(f"Created prediction DataFrame with {len(output_df.columns)} columns")
    print(f"Columns: {list(output_df.columns)}")
    
    return output_df

def save_domain_models(models_dict, feature_groups, output_dir='domain_models'):
    """
    Save trained domain models to disk
    
    Args:
        models_dict: Dictionary of trained models
        feature_groups: Dictionary of feature groups for each domain
        output_dir: Directory to save models
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save each model
    ensemble_weights = {'domains': {}}
    
    for domain, model_info in models_dict.items():
        model = model_info.get('model')
        if model is None:
            print(f"Warning: No model found for {domain}, skipping")
            continue
            
        score = model_info.get('score', 0.0)
        weight = model_info.get('weight', 0.0)
        
        # Get features for this domain
        features = feature_groups.get(domain, [])
        
        # Save model
        model_path = os.path.join(output_dir, f"{domain}_model.joblib")
        try:
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'domain': domain,
                'features': features,
                'score': float(score),
                'weight': float(weight),
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = os.path.join(output_dir, f"{domain}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            # Add to ensemble weights
            ensemble_weights['domains'][domain] = float(weight)
            
            print(f"Saved model and metadata for {domain}")
        except Exception as e:
            print(f"Error saving model for {domain}: {e}")
    
    # Save ensemble weights
    ensemble_path = os.path.join(output_dir, 'ensemble_weights.json')
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_weights, f, indent=4)
    
    print(f"Saved ensemble weights for {len(ensemble_weights['domains'])} domains")

def score_predictions_with_numerai_tools(predictions_df, target_col='target', era_col='era'):
    """
    Score predictions using numerai-tools if available
    
    Args:
        predictions_df: DataFrame with predictions and targets
        target_col: Target column name
        era_col: Era column name
        
    Returns:
        dict: Dictionary with scoring results
    """
    if not NUMERAI_TOOLS_AVAILABLE:
        print("numerai-tools not available, using basic scoring")
        return None
    
    try:
        # First, check if the era column exists in the DataFrame
        if era_col not in predictions_df.columns:
            print(f"Error: '{era_col}' column not found in predictions DataFrame")
            print(f"Available columns: {list(predictions_df.columns)}")
            return None
            
        # Check if target_col exists
        if target_col not in predictions_df.columns:
            print(f"Error: '{target_col}' column not found in predictions DataFrame")
            print(f"Available columns: {list(predictions_df.columns)}")
            return None
            
        results = {}
        
        # Get prediction columns (those starting with 'pred_' or 'prediction_')
        pred_cols = [col for col in predictions_df.columns if col.startswith('pred_') or col.startswith('prediction_')]
        
        if not pred_cols:
            print("No prediction columns found in DataFrame")
            return None
            
        # Create a DataFrame with just the prediction columns for correlation_contribution
        # THE FIX: correlation_contribution expects a DataFrame for predictions
        # not numpy arrays or a single Series
        preds_df = predictions_df[pred_cols]
        
        # Get meta-model (use first prediction column as meta-model)
        # THE FIX: correlation_contribution expects a pandas Series for meta_model
        # not a numpy array
        meta_model = predictions_df[pred_cols[0]]
        
        # Get targets
        # THE FIX: correlation_contribution expects a pandas Series for live_targets
        # not a numpy array
        targets = predictions_df[target_col]
        
        # Calculate correlation contribution
        try:
            # THE FIX: This is the key part that fixes the error in the traceback
            # We pass pandas objects (DataFrame, Series, Series) instead of numpy arrays
            # The error occurred because numpy arrays don't have a .dropna() method
            # which is called by filter_sort_index() inside correlation_contribution()
            cc = correlation_contribution(
                preds_df,  # Pass DataFrame of predictions (not .values)
                meta_model,  # Pass meta-model Series (not .values)
                targets  # Pass targets Series (not .values)
            )
            
            # Process results for each prediction column
            for pred_col in pred_cols:
                # Get correlation by era
                # Calculate per-era correlation (requires at least 5 data points per era)
                pred_by_era = predictions_df.groupby(era_col).apply(
                    lambda x: x[pred_col].corr(x[target_col]) if len(x) > 5 else np.nan
                ).dropna()
                
                # Calculate metrics
                corr_mean = pred_by_era.mean()
                corr_std = pred_by_era.std()
                sharpe = corr_mean / corr_std if corr_std > 0 else 0
                
                # Store metrics
                results[pred_col] = {
                    'mean_correlation': corr_mean,
                    'std_correlation': corr_std,
                    'sharpe_ratio': sharpe,
                    'feature_exposure': cc.get(pred_col, 0),
                    'correlation_by_era': pred_by_era.to_dict()
                }
            
            print("Scoring with numerai-tools completed")
            return results
            
        except Exception as inner_e:
            # Catch errors specifically in the correlation_contribution calculation
            print(f"Error in correlation_contribution calculation: {inner_e}")
            traceback.print_exc()
            raise inner_e
            
    except Exception as e:
        # This catches the error shown in the traceback and prints it
        print(f"Error in scoring with numerai-tools: {e}")
        traceback.print_exc()
        
        # Fallback to basic correlation scoring
        print("Falling back to basic correlation scoring...")
        return None

def run_integrated_pipeline(data_version="v5.0", 
                            feature_set="medium",
                            sample_size=100000,
                            domain_score_threshold=0.05,
                            correlation_threshold=0.95,
                            tournament_data_path=None,
                            models_dir='domain_models'):
    """
    Run the integrated pipeline from domain identification to predictions
    
    Args:
        data_version: Numerai data version
        feature_set: Feature set to use (small, medium, all)
        sample_size: Number of samples to use
        domain_score_threshold: Min threshold for domain scores
        correlation_threshold: Threshold for feature correlation pruning
        tournament_data_path: Path to tournament data file for predictions
        models_dir: Directory to save/load models
        
    Returns:
        dict: Results from the pipeline
    """
    # Start with clean memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    log_memory_usage("Starting pipeline")
    
    results = {
        'domains_identified': 0,
        'models_trained': 0,
        'models_saved': True,
        'validation_score': None,
        'tournament_predictions': None
    }
    
    # Step 1: Run domains pipeline to identify feature groups
    print("\n=== Phase 1: Domain Identification ===")
    domains_results = run_domains_only_pipeline(
        data_version=data_version,
        feature_set=feature_set,
        main_target="target",
        aux_targets=["target_cyrusd_20", "target_cyrusd_60", "target_teager2b_20", "target_teager2b_60"],
        sample_size=sample_size,
        n_clusters=None,  # Auto-determine optimal clusters
        use_incremental=True,
        skip_visualizations=True,
        random_seed=42
    )
    
    if not domains_results or 'feature_groups' not in domains_results:
        print("Domain identification failed")
        return results
    
    feature_groups = domains_results['feature_groups']
    results['domains_identified'] = len(feature_groups)
    print(f"Identified {results['domains_identified']} domains")
    
    # Log memory usage
    log_memory_usage("After domain identification")
    
    # Step 2: Run follow-up pipeline to train models
    print("\n=== Phase 2: Model Training ===")
    
    # Clean memory before loading data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load data for training
    train_df, val_df, features, all_targets = load_data(
        data_version=data_version,
        feature_set=feature_set,
        main_target="target",
        num_aux_targets=3,
        sample_size=sample_size,
        random_seed=42
    )
    
    if train_df is None or val_df is None:
        print("Failed to load training/validation data")
        return results
    
    log_memory_usage("After data loading")
    
    # Run follow-up pipeline
    followup_results = follow_up_domains_pipeline(
        train_df=train_df,
        val_df=val_df,
        feature_groups=feature_groups,
        main_target="target",
        domain_score_threshold=domain_score_threshold,
        correlation_threshold=correlation_threshold
    )
    
    if not followup_results or 'domain_models' not in followup_results:
        print("Model training failed")
        return results
    
    domain_models = followup_results['domain_models']
    results['models_trained'] = len(domain_models)
    print(f"Trained {results['models_trained']} domain models")
    
    # Save the final score
    if 'final_model_score' in followup_results:
        results['validation_score'] = followup_results['final_model_score']
        print(f"Validation score: {results['validation_score']:.4f}")
    
    # Step 3: Save models
    print("\n=== Phase 3: Saving Models ===")
    try:
        save_domain_models(
            models_dict=domain_models,
            feature_groups=feature_groups,
            output_dir=models_dir
        )
    except Exception as e:
        print(f"Error saving models: {e}")
        results['models_saved'] = False
    
    # Step 4: Make tournament predictions if path provided
    if tournament_data_path:
        print("\n=== Phase 4: Tournament Predictions ===")
        try:
            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tournament data
            print(f"Loading tournament data from {tournament_data_path}")
            tournament_data = pd.read_parquet(tournament_data_path)
            print(f"Loaded tournament data with shape {tournament_data.shape}")
            
            # Convert features to float32
            for col in tournament_data.columns:
                if col in features:
                    tournament_data[col] = tournament_data[col].astype(np.float32)
            
            # Make predictions
            predictions = make_predictions_pipeline(
                tournament_data, 
                models_dir=models_dir
            )
            
            if predictions is not None:
                # Format for submission
                if 'prediction_ensemble' in predictions.columns:
                    predictions['prediction'] = predictions['prediction_ensemble']
                elif 'prediction_best' in predictions.columns:
                    predictions['prediction'] = predictions['prediction_best']
                
                # Keep only required columns
                id_cols = ['id', 'era'] if 'id' in tournament_data.columns else ['era']
                submission = predictions[id_cols + ['prediction']]
                
                # Save submission
                submission_path = 'numerai_submission.csv'
                submission.to_csv(submission_path, index=False)
                
                print(f"Saved tournament predictions to {submission_path}")
                results['tournament_predictions'] = submission_path
            else:
                print("Failed to generate predictions")
        except Exception as e:
            print(f"Error making tournament predictions: {e}")
            traceback.print_exc()
    
    # Calculate total runtime
    total_time = (time.time() - start_time) / 60  # minutes
    print(f"\nPipeline completed in {total_time:.2f} minutes")
    
    # Log final memory usage
    log_memory_usage("End of pipeline")
    
    return results

def score_models_with_validation(models_dir='domain_models', data_version="v5.0", 
                                feature_set="medium", sample_size=100000):
    """
    Score saved models using validation data
    
    Args:
        models_dir: Directory containing saved models
        data_version: Numerai data version
        feature_set: Feature set to use
        sample_size: Sample size for validation
        
    Returns:
        dict: Scoring results
    """
    print("\n=== Scoring Models with Validation Data ===")
    
    # Load validation data
    train_df, val_df, _, _ = load_data(
        data_version=data_version,
        feature_set=feature_set,
        main_target="target",
        num_aux_targets=3,
        sample_size=sample_size * 2,  # Use more data for validation
        random_seed=42
    )
    
    if val_df is None:
        print("Failed to load validation data")
        return None
    
    # Verify 'era' column exists
    if 'era' not in val_df.columns:
        print("Warning: 'era' column not found in validation data")
        print(f"Available columns: {list(val_df.columns)}")
        print("Adding dummy 'era' column for compatibility")
        # Add a dummy era column if needed (e.g., all rows in one era)
        val_df['era'] = 1
    
    # Make predictions
    predictions = make_predictions_pipeline(
        val_df, 
        target_col="target", 
        models_dir=models_dir
    )
    
    if predictions is None:
        print("Failed to generate predictions")
        return None
    
    # Verify columns in predictions DataFrame
    print(f"Prediction columns: {list(predictions.columns)}")
    
    # Ensure target column is present
    if 'target' not in predictions.columns and 'target' in val_df.columns:
        print("Adding target column to predictions DataFrame")
        predictions['target'] = val_df['target']
    
    # Make sure era column is present for proper scoring
    if 'era' not in predictions.columns:
        print("Warning: 'era' column missing from predictions, adding from validation data")
        if 'era' in val_df.columns:
            predictions['era'] = val_df['era']
        else:
            print("Creating a dummy 'era' column (all rows in one era)")
            predictions['era'] = 1
    
    # Score with numerai-tools if available
    if NUMERAI_TOOLS_AVAILABLE:
        print("Scoring with numerai-tools...")
        try:
            # Check that we have both the target and era columns
            if 'target' not in predictions.columns:
                print("Error: 'target' column not found in predictions DataFrame")
                raise ValueError("Missing target column")
                
            if 'era' not in predictions.columns:
                print("Error: 'era' column not found in predictions DataFrame")
                raise ValueError("Missing era column")
            
            # Get prediction columns
            pred_cols = [col for col in predictions.columns 
                        if col.startswith('pred_') or col.startswith('prediction_')]
            
            if not pred_cols:
                print("Error: No prediction columns found")
                raise ValueError("No prediction columns")
                
            # Use our fixed scoring function
            scoring_results = score_predictions_with_numerai_tools(
                predictions_df=predictions,
                target_col="target",
                era_col="era"
            )
            
            if scoring_results:
                # Save results to JSON
                scores_path = os.path.join(models_dir, 'validation_scores.json')
                with open(scores_path, 'w') as f:
                    # Convert numpy types to Python native types for JSON
                    json_results = {}
                    for key, value in scoring_results.items():
                        # Skip correlation_by_era which can be large and may contain non-serializable types
                        json_results[key] = {
                            k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                            for k, v in value.items() if k != 'correlation_by_era'
                        }
                    
                    json.dump(json_results, f, indent=4)
                
                print(f"Saved validation scores to {scores_path}")
                return scoring_results
        except Exception as e:
            print(f"Error in numerai-tools scoring: {e}")
            traceback.print_exc()
            print("Falling back to basic correlation scoring...")
    else:
        print("numerai-tools not available, using basic correlation scoring")
    
    # Basic correlation scoring as fallback
    print("Using basic correlation scoring...")
    
    # Check if 'era' exists for grouping
    if 'era' in predictions.columns:
        eras = predictions['era'].unique()
    else:
        # Create a single era if 'era' column is missing
        eras = [1]
        predictions['era'] = 1
    
    # Calculate metrics per era
    results = []
    for era in eras:
        era_data = predictions[predictions['era'] == era]
        
        if len(era_data) == 0:
            continue
            
        era_result = {'era': era}
        
        # Calculate metrics for each prediction column
        for col in era_data.columns:
            if col.startswith('pred_') or col.startswith('prediction_'):
                # Calculate correlation with target
                if 'target' in era_data.columns:
                    # Use pandas correlation method to avoid potential issues with np.corrcoef
                    try:
                        # Remove NaN values
                        valid_mask = ~np.isnan(era_data[col]) & ~np.isnan(era_data['target'])
                        if valid_mask.sum() >= 10:  # Need at least 10 valid pairs
                            corr = era_data[col][valid_mask].corr(era_data['target'][valid_mask])
                            era_result[f"{col}_corr"] = corr
                    except Exception as calc_err:
                        print(f"Error calculating correlation for {col} in era {era}: {calc_err}")
        
        results.append(era_result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {}
    prediction_cols = [col for col in results_df.columns if col.endswith('_corr')]
    
    for col in prediction_cols:
        model_name = col.replace('_corr', '')
        # Handle empty or all-NaN columns
        if col in results_df.columns and not results_df[col].isna().all():
            mean_corr = results_df[col].mean()
            std_corr = results_df[col].std()
            
            summary[model_name] = {
                'mean_correlation': mean_corr,
                'median_correlation': results_df[col].median(),
                'std_correlation': std_corr,
                'min_correlation': results_df[col].min(),
                'max_correlation': results_df[col].max(),
                'sharpe_ratio': mean_corr / std_corr if std_corr > 0 else 0,
                'positive_eras': (results_df[col] > 0).mean() * 100  # Percentage of positive eras
            }
    
    # Save summary to file
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(os.path.join(models_dir, 'backtest_summary.csv'))
    
    # Save detailed era results
    results_df.to_csv(os.path.join(models_dir, 'backtest_by_era.csv'), index=False)
    
    # Print summary
    print("\nScoring Summary:")
    for model_name, metrics in sorted(summary.items(), 
                                     key=lambda x: x[1]['mean_correlation'], 
                                     reverse=True)[:5]:
        print(f"{model_name}:")
        print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    print(f"Saved basic scoring results to {models_dir}/backtest_summary.csv")
    
    return {'summary': summary, 'by_era': results_df}

if __name__ == "__main__":
    # Define parameters
    DATA_VERSION = "v5.0"
    FEATURE_SET = "medium"  # Options: small, medium, all
    SAMPLE_SIZE = 100000    # Adjust based on your memory constraints
    DOMAIN_SCORE_THRESHOLD = 0.05
    CORRELATION_THRESHOLD = 0.95
    MODELS_DIR = "domain_models"
    TOURNAMENT_PATH = None  # Set to tournament.parquet path if available
    
    # Run the full pipeline
    results = run_integrated_pipeline(
        data_version=DATA_VERSION,
        feature_set=FEATURE_SET,
        sample_size=SAMPLE_SIZE,
        domain_score_threshold=DOMAIN_SCORE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        tournament_data_path=TOURNAMENT_PATH,
        models_dir=MODELS_DIR
    )
    
    # Score models with validation data
    if results['models_saved']:
        print("\n=== Scoring Models with Validation Data ===")
        scoring_results = score_models_with_validation(
            models_dir=MODELS_DIR,
            data_version=DATA_VERSION,
            feature_set=FEATURE_SET,
            sample_size=min(SAMPLE_SIZE * 2, 200000)  # Use more data for scoring if possible
        )
        
        if scoring_results:
            print("\nScoring Summary:")
            
            # If using numerai-tools
            if isinstance(scoring_results, dict) and 'summary' not in scoring_results:
                # Print top models by correlation
                top_models = sorted(scoring_results.items(), 
                                    key=lambda x: x[1]['mean_correlation'], 
                                    reverse=True)[:5]
                
                for model, metrics in top_models:
                    print(f"{model}:")
                    print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            
            # If using basic scoring
            elif isinstance(scoring_results, dict) and 'summary' in scoring_results:
                top_models = sorted(scoring_results['summary'].items(), 
                                   key=lambda x: x[1]['mean_correlation'], 
                                   reverse=True)[:5]
                
                for model, metrics in top_models:
                    print(f"{model}:")
                    print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
                    print(f"  Sharpe Ratio: {metrics['sharpe']:.4f}")