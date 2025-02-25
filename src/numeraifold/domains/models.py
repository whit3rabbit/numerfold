def load_domain_models(models_dir='domain_models'):
    """
    Load all saved domain models and their metadata
    
    Args:
        models_dir: Directory where domain models are saved
        
    Returns:
        dict: Dictionary of domain models and their metadata
    """
    import os
    import joblib
    import json
    
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
    import numpy as np
    import pandas as pd
    
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
    import pandas as pd
    import numpy as np
    
    # Load domain models
    domain_models = load_domain_models(models_dir)
    if not domain_models:
        print("No domain models found, cannot make predictions")
        return None
    
    # Make predictions
    results = predict_with_domain_models(new_data, domain_models, target_col)
    
    # Create output DataFrame
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
    
    # Add target for reference if available
    if target_col is not None and target_col in new_data.columns:
        output_df[target_col] = new_data[target_col]
    
    # Add metadata
    print(f"Created prediction DataFrame with {len(output_df.columns)} columns")
    
    return output_df