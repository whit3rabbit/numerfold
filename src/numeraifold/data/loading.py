"""
loading.py

This module provides functions to load and process data for a machine learning pipeline.
It includes functions to download datasets if missing, load data with memory-efficient
settings, process data in batches for model inference, and create properly formatted model inputs.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pyarrow.parquet as pq
import gc
from numerapi import NumerAPI

from numeraifold.utils.logging import log_memory_usage

def load_data(data_version="v5.0", feature_set="small",
              main_target="target", aux_targets=None, num_aux_targets=5):
    """
    Load data with memory-efficient settings and robust target handling.
    """
    print(f"Loading {data_version} data with {feature_set} feature set...")
    
    # Download data if needed
    napi = NumerAPI()
    for file in ['train.parquet', 'validation.parquet', 'features.json']:
        filepath = f"{data_version}/{file}"
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            napi.download_dataset(filepath)

    # Load feature metadata
    with open(f"{data_version}/features.json") as f:
        feature_metadata = json.load(f)
    features = feature_metadata["feature_sets"][feature_set]

    # Get available targets from schema without printing
    schema = pq.read_schema(f"{data_version}/train.parquet")
    available_targets = [col for col in schema.names if col.startswith('target')]
    
    # Validate all targets before loading
    if main_target not in available_targets:
        print(f"Warning: {main_target} not found in dataset. Using first available target.")
        main_target = available_targets[0] if available_targets else None
        
    if not main_target:
        print("No valid main target found. Please check the dataset.")
        return None, None, features, []

    # Validate aux_targets if provided
    final_targets = [main_target]
    if aux_targets is not None:
        missing_targets = [t for t in aux_targets if t not in available_targets]
        if missing_targets:
            print(f"Error: The following auxiliary targets are not available: {missing_targets}")
            print("Available targets:", [t for t in available_targets if t.startswith('target_')])
            return None, None, features, []
        final_targets.extend([t for t in aux_targets if t != main_target])
    else:
        # Use random auxiliary targets
        remaining_targets = [t for t in available_targets if t != main_target]
        if num_aux_targets > 0:
            num_to_add = min(num_aux_targets, len(remaining_targets))
            if num_to_add < num_aux_targets:
                print(f"Warning: Only {num_to_add} additional targets available")
            selected = np.random.choice(remaining_targets, size=num_to_add, replace=False)
            final_targets.extend(selected)

    print(f"Using targets: {final_targets}")
    
    # Load data with memory efficiency
    try:
        columns_to_load = ["era"] + features + final_targets
        print(f"Reading train data with {len(features)} features and {len(final_targets)} targets...")
        
        train_df = pd.read_parquet(
            f"{data_version}/train.parquet",
            columns=columns_to_load,
            dtype_backend='pyarrow'
        )
        
        print("Reading validation data...")
        val_df = pd.read_parquet(
            f"{data_version}/validation.parquet",
            columns=columns_to_load,
            dtype_backend='pyarrow'
        )
        
        print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
        return train_df, val_df, features, final_targets
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, features, []

def process_in_batches(df, model, features, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process the input DataFrame in batches to perform model inference.
    
    Parameters:
        df (pd.DataFrame): Input data with feature columns.
        model (torch.nn.Module): PyTorch model for inference.
        features (list): List of feature column names to process.
        device (str or torch.device): Device on which to run the model.
    
    Returns:
        tuple: A tuple containing:
            - embeddings (np.ndarray): Pooled feature embeddings for each sample.
            - confidences (np.ndarray): Confidence scores for each prediction.
            - predictions (np.ndarray): Model predictions for each sample.
    """
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) < len(features):
        print(f"Warning: {len(features) - len(numeric_features)} non-numeric features were excluded")
        features = numeric_features

    df = df[features].convert_dtypes()

    all_embeddings = []
    all_confidences = []
    all_predictions = []

    batch_size = 128
    n_batches = max(1, len(df) // batch_size)
    batches = np.array_split(df.index, n_batches)

    for batch_idx in tqdm(batches, desc="Processing batches"):
        try:
            batch_df = df.loc[batch_idx]

            try:
                batch_data = batch_df.astype(float).values.astype(np.float32)
            except Exception as e:
                print(f"Error converting batch to float: {e}")
                batch_data = batch_df.fillna(0).astype(float).values.astype(np.float32)

            X_batch = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(-1)
            X_batch = X_batch.to(device)

            with torch.no_grad():
                try:
                    predictions, attention_outputs = model(X_batch)
                    x = model.feature_embedding(X_batch)

                    for transformer in model.transformer_blocks:
                        x, _ = transformer(x)

                    x = model.output_norm(x)
                    pooled_features = x.mean(dim=1).cpu().numpy()

                    if isinstance(attention_outputs, list) and len(attention_outputs) > 0:
                        attention_outputs = [attn for attn in attention_outputs if attn is not None]
                        if attention_outputs:
                            attention_stack = torch.stack(attention_outputs)
                            attention_mean = attention_stack.mean(dim=[0, 2])
                            attention_entropy = -(attention_mean * torch.log(attention_mean + 1e-10)).sum(dim=-1).mean(dim=-1)
                            confidence = 1.0 / (1.0 + attention_entropy.cpu().numpy())
                        else:
                            confidence = np.ones(len(predictions)) * 0.5
                    else:
                        confidence = np.ones(len(predictions)) * 0.5

                    predictions = predictions.cpu().numpy()
                    if len(predictions.shape) > 1:
                        predictions = predictions.reshape(-1)

                    assert len(predictions) == len(confidence) == len(pooled_features), (
                        f"Shape mismatch: predictions {predictions.shape}, confidence {confidence.shape}, "
                        f"features {pooled_features.shape}"
                    )

                    all_embeddings.append(pooled_features)
                    all_confidences.append(confidence)
                    all_predictions.append(predictions)

                except Exception as e:
                    print(f"Error in model forward pass: {str(e)}")
                    raise e

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            batch_size = len(batch_idx)
            dummy_embeddings = np.zeros((batch_size, model.embed_dim))
            dummy_confidences = np.ones(batch_size) * 0.5
            dummy_predictions = np.ones(batch_size) * batch_df.mean().mean()

            all_embeddings.append(dummy_embeddings)
            all_confidences.append(dummy_confidences)
            all_predictions.append(dummy_predictions)

    try:
        embeddings = np.vstack(all_embeddings)
        confidences = np.concatenate(all_confidences)
        predictions = np.concatenate(all_predictions)

        print(f"Final shapes - embeddings: {embeddings.shape}, confidences: {confidences.shape}, predictions: {predictions.shape}")

        if len(predictions.shape) > 1:
            predictions = predictions.reshape(-1)

        return embeddings, confidences, predictions
    except Exception as e:
        print(f"Error combining results: {str(e)}")
        print("Shapes of accumulated results:")
        print(f"embeddings: {[emb.shape for emb in all_embeddings]}")
        print(f"confidences: {[conf.shape for conf in all_confidences]}")
        print(f"predictions: {[pred.shape for pred in all_predictions]}")
        return None, None, None


def create_model_inputs(df, features, era_col='era', dtype=np.float32):
    """
    Create model inputs by converting feature columns to numeric and reshaping the data.
    Handles non-numeric columns by attempting conversion and fills missing values with zero.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing feature columns.
        features (list): List of feature column names to extract.
        era_col (str): Column name for era information (unused in conversion, but may be relevant).
        dtype (data-type): Desired numpy data type for the output array.
    
    Returns:
        np.ndarray: A 3D numpy array of shape [n_samples, n_features, 1] ready for model input.
    """
    non_numeric_cols = []
    for col in features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"Converting non-numeric columns to float: {non_numeric_cols}")
        df_copy = df.copy()
        for col in non_numeric_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    else:
        df_copy = df

    feature_data = df_copy[features].fillna(0).values

    if feature_data.dtype == object:
        print("Converting object array to float32...")
        numeric_data = np.zeros(feature_data.shape, dtype=dtype)
        for i in range(feature_data.shape[0]):
            for j in range(feature_data.shape[1]):
                try:
                    val = feature_data[i, j]
                    numeric_data[i, j] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    numeric_data[i, j] = 0.0
        feature_data = numeric_data

    feature_data = feature_data.reshape(feature_data.shape[0], feature_data.shape[1], 1)

    return feature_data


def optimize_dataframe(df, features=None, target=None, inplace=False):
    """
    Optimize a pandas DataFrame's memory usage by converting to appropriate dtypes.
    
    Args:
        df: DataFrame to optimize
        features: List of feature column names to convert to float32
        target: Target column name to convert to float32
        inplace: Whether to modify the DataFrame in place
        
    Returns:
        Optimized DataFrame
    """
    if not inplace:
        df = df.copy()
    
    # Convert specific features to float32
    if features is not None:
        for col in features:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
    
    # Convert target to float32
    if target is not None:
        if isinstance(target, list):
            for t in target:
                if t in df.columns:
                    df[t] = df[t].astype(np.float32)
        elif target in df.columns:
            df[target] = df[target].astype(np.float32)
    
    # Optimize all other numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        # Skip already processed columns
        if features is not None and col in features:
            continue
        if target is not None:
            if isinstance(target, list) and col in target:
                continue
            elif col == target:
                continue
        
        # Convert float64 to float32
        df[col] = df[col].astype(np.float32)
    
    # Convert integer columns to appropriate size
    for col in df.select_dtypes(include=['int64']).columns:
        # Check if int32 range is sufficient
        if df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    
    # Log memory savings
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Optimized DataFrame memory usage: {memory_usage_mb:.2f} MB")
    
    return df

def chunk_process_dataframe(df, chunk_size=10000, func=None):
    """
    Process a large DataFrame in chunks to avoid memory issues.
    
    Args:
        df: DataFrame to process
        chunk_size: Size of each chunk
        func: Function to apply to each chunk, should take a DataFrame as input
              and return a DataFrame
    
    Returns:
        Processed DataFrame (combined result of processing all chunks)
    """
    if func is None:
        return df
    
    result_chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        print(f"Processing chunk {i//chunk_size + 1}/{(len(df) + chunk_size - 1)//chunk_size}")
        
        # Apply function to chunk
        processed_chunk = func(chunk)
        result_chunks.append(processed_chunk)
        
        # Free memory
        del chunk
        gc.collect()
    
    # Combine results
    result = pd.concat(result_chunks, axis=0, ignore_index=True)
    return result

def load_and_optimize(file_path, features=None, target=None, chunk_size=None):
    """
    Load a CSV or Parquet file and optimize its memory usage.
    For large files, the loading and optimization is done in chunks.
    
    Args:
        file_path: Path to the CSV or Parquet file
        features: List of feature column names
        target: Target column name or list of names
        chunk_size: Size of chunks to use for loading large files
                    If None, the file is loaded in one go
    
    Returns:
        Optimized DataFrame
    """
    log_memory_usage("Before loading file")
    
    # Determine file type
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if chunk_size is not None:
        # Load in chunks
        if file_extension == '.csv':
            chunks = pd.read_csv(file_path, chunksize=chunk_size)
        elif file_extension in ['.parquet', '.pq']:
            # Custom chunk loading for parquet
            pq_file = pq.ParquetFile(file_path)
            chunks = []
            for i in range(0, pq_file.metadata.num_rows, chunk_size):
                chunk = pd.read_parquet(
                    file_path,
                    skip_rows=i,
                    num_rows=min(chunk_size, pq_file.metadata.num_rows - i)
                )
                chunks.append(chunk)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Process chunks
        optimized_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Optimizing chunk {i+1}")
            optimized_chunk = optimize_dataframe(chunk, features, target)
            optimized_chunks.append(optimized_chunk)
            
            # Free memory
            del chunk
            gc.collect()
        
        # Combine results
        result = pd.concat(optimized_chunks, axis=0, ignore_index=True)
        del optimized_chunks
        gc.collect()
        
    else:
        # Load in one go
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Optimize
        result = optimize_dataframe(df, features, target)
    
    log_memory_usage("After loading and optimizing file")
    return result