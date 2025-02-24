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
import pyarrow as pa
from numerapi import NumerAPI
import gc


def load_data(data_version="v5.0", feature_set="small",
              main_target="target", aux_targets=None, num_aux_targets=5,
              convert_dtypes=True, chunk_size=None, max_rows=None):
    """
    Load data with memory-efficient settings and robust target handling.
    
    Parameters:
        data_version (str): Identifier for the data version/folder
        feature_set (str): Feature set key to use ('small', 'medium', 'all')
        main_target (str): Primary target column name
        aux_targets (list): Optional list of specific auxiliary targets to include
        num_aux_targets (int): Number of random auxiliary targets to include if aux_targets not specified
        convert_dtypes (bool): If True, converts PyArrow dtypes to standard Python/NumPy types
        chunk_size (int): If provided, process data in chunks of this size to save memory
        max_rows (int): If provided, limit the number of rows loaded
    
    Returns:
        tuple: (train_df, val_df, features, all_targets)
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

    # Get schema and available targets
    schema = pq.read_schema(f"{data_version}/train.parquet")
    all_columns = schema.names
    available_targets = [col for col in all_columns if col.startswith('target')]
    
    # Validate and process main target
    if main_target not in available_targets:
        print(f"Warning: {main_target} not found in dataset. Using first available target.")
        main_target = available_targets[0] if available_targets else None
        
    if not main_target:
        print("No valid main target found. Please check the dataset.")
        return None, None, features, []

    # Build final target list
    if aux_targets is not None:
        # Use specified auxiliary targets
        final_targets = [main_target] + [t for t in aux_targets if t != main_target]
    else:
        # Use random auxiliary targets
        remaining_targets = [t for t in available_targets if t != main_target]
        if num_aux_targets > 0:
            num_to_add = min(num_aux_targets, len(remaining_targets))
            if num_to_add < num_aux_targets:
                print(f"Warning: Only {num_to_add} additional targets available")
            selected = np.random.choice(remaining_targets, size=num_to_add, replace=False)
            final_targets = [main_target] + list(selected)
        else:
            final_targets = [main_target]

    print(f"Using targets: {final_targets}")
    
    # Load data with memory efficiency
    columns_to_load = ["era"] + features + final_targets
    print(f"Reading train data with {len(features)} features and {len(final_targets)} targets...")
    
    # If chunk processing is enabled, use it
    if chunk_size is not None:
        print(f"Loading and processing data in chunks of {chunk_size} rows")
        train_df = load_data_in_chunks(
            filepath=f"{data_version}/train.parquet",
            columns=columns_to_load,
            chunk_size=chunk_size,
            max_rows=max_rows,
            convert_dtypes=convert_dtypes
        )
        
        print("Reading validation data in chunks...")
        val_df = load_data_in_chunks(
            filepath=f"{data_version}/validation.parquet",
            columns=columns_to_load,
            chunk_size=chunk_size,
            max_rows=max_rows,
            convert_dtypes=convert_dtypes
        )
    else:       
        def cast_table_to_float32(table):
            """Cast numeric columns to float32 in PyArrow table"""
            for col in table.column_names:
                if col == 'era':
                    continue
                field = table.schema.field(col)
                field_type = field.type
                if (pa.types.is_integer(field_type) or 
                    pa.types.is_floating(field_type) or 
                    pa.types.is_decimal(field_type)):
                    table = table.set_column(
                        table.column_names.index(col),
                        col,
                        table[col].cast(pa.float32())
                    )
            return table

        # Load and process training data
        table = pq.read_table(f"{data_version}/train.parquet", columns=columns_to_load)
        if convert_dtypes:
            table = cast_table_to_float32(table)
        train_df = table.to_pandas(dtype_backend='pyarrow' if not convert_dtypes else 'numpy')
        
        if max_rows and len(train_df) > max_rows:
            train_df = train_df.head(max_rows)

        # Load validation data
        table_val = pq.read_table(f"{data_version}/validation.parquet", columns=columns_to_load)
        if convert_dtypes:
            table_val = cast_table_to_float32(table_val)
        val_df = table_val.to_pandas(dtype_backend='pyarrow' if not convert_dtypes else 'numpy')
        
        if max_rows and len(val_df) > max_rows:
            val_df = val_df.head(max_rows)

    # Post-processing for chunked data
    if chunk_size is not None and convert_dtypes:
        def bulk_convert(df):
            """Optimized bulk conversion for chunked data"""
            numeric_cols = df.select_dtypes(include=np.number).columns.difference(['era'])
            if not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].astype('float32')
            return df
        
        print("Optimizing dtype conversion for chunked data...")
        train_df = bulk_convert(train_df)
        val_df = bulk_convert(val_df)
        gc.collect()

    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
    return train_df, val_df, features, final_targets


def load_data_in_chunks(filepath, columns, chunk_size=10000, max_rows=None, convert_dtypes=True):
    """Optimized chunked loading with PyArrow type conversion"""
    parquet_file = pq.ParquetFile(filepath)
    total_rows = parquet_file.metadata.num_rows
    if max_rows:
        total_rows = min(total_rows, max_rows)

    chunks = []
    for i in range(0, total_rows, chunk_size):
        row_group = i // parquet_file.metadata.row_group(0).num_rows
        if row_group >= parquet_file.num_row_groups:
            break

        table = parquet_file.read_row_group(row_group, columns=columns)
        if convert_dtypes:
            for col in table.column_names:
                if col == 'era':
                    continue
                field = table.schema.field(col)
                field_type = field.type
                if (pa.types.is_integer(field_type) or 
                    pa.types.is_floating(field_type) or 
                    pa.types.is_decimal(field_type)):
                    table = table.set_column(
                        table.column_names.index(col),
                        col,
                        table[col].cast(pa.float32())
        
        df = table.to_pandas()
        chunks.append(df.iloc[:chunk_size])
        
        if max_rows and sum(len(c) for c in chunks) >= max_rows:
            break

    result = pd.concat(chunks, ignore_index=True)
    return result.head(max_rows) if max_rows else result

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
