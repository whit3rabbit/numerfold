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
    Optimized for minimal memory usage through schema-aware loading and batch processing.
    
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
        final_targets = [main_target] + [t for t in aux_targets if t != main_target]
    else:
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
    
    # Create optimized schema for dtype conversion
    if convert_dtypes:
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

    # Memory-optimized loading strategy
    if chunk_size is not None:
        print(f"Loading and processing data in chunks of {chunk_size} rows")
        load_params = {
            "filepath": f"{data_version}/train.parquet",
            "columns": columns_to_load,
            "chunk_size": chunk_size,
            "max_rows": max_rows,
            "convert_dtypes": convert_dtypes,
            "optimized_schema": optimized_schema
        }
        train_df = load_data_in_chunks(**load_params)
        
        print("Reading validation data in chunks...")
        load_params["filepath"] = f"{data_version}/validation.parquet"
        val_df = load_data_in_chunks(**load_params)
    else:
        # Single-pass optimized loading
        read_params = {
            "columns": columns_to_load,
            "schema": optimized_schema if convert_dtypes else None
        }
        
        # Handle max_rows efficiently for non-chunked loading
        if max_rows:
            train_table = read_table_with_row_cap(f"{data_version}/train.parquet", max_rows, **read_params)
            val_table = read_table_with_row_cap(f"{data_version}/validation.parquet", max_rows, **read_params)
        else:
            train_table = pq.read_table(f"{data_version}/train.parquet", **read_params)
            val_table = pq.read_table(f"{data_version}/validation.parquet", **read_params)
        
        train_df = train_table.to_pandas()
        val_df = val_table.to_pandas()

    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
    return train_df, val_df, features, final_targets


def load_data_in_chunks(filepath, columns, chunk_size=10000, max_rows=None,
                        convert_dtypes=True, optimized_schema=None):
    """Memory-optimized chunked loading using PyArrow's batch-wise reading"""
    parquet_file = pq.ParquetFile(filepath)
    total_rows = min(parquet_file.metadata.num_rows, max_rows) if max_rows else parquet_file.metadata.num_rows
    
    chunks = []
    rows_processed = 0

    # Check if iter_batches is available in the PyArrow version
    if hasattr(parquet_file, 'iter_batches'):
        # Modern PyArrow with streaming API
        for batch in parquet_file.iter_batches(
            columns=columns,
            batch_size=chunk_size,
            schema=optimized_schema if convert_dtypes else None
        ):
            if rows_processed >= total_rows:
                break
                
            # Convert batch to pandas
            df = batch.to_pandas()
            
            # Handle remaining rows if we're near max_rows
            remaining = total_rows - rows_processed
            if len(df) > remaining:
                df = df.iloc[:remaining]
            
            chunks.append(df)
            rows_processed += len(df)
    else:
        # Fallback for older PyArrow versions
        for row_group in range(parquet_file.num_row_groups):
            if rows_processed >= total_rows:
                break
                
            # Read row group
            batch = parquet_file.read_row_group(row_group, columns=columns)
            
            # Apply optimized schema conversion if needed
            if convert_dtypes and optimized_schema:
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
            
            df = batch.to_pandas()
            
            # Process in smaller chunks if row group is larger than chunk_size
            start_idx = 0
            while start_idx < len(df):
                end_idx = min(start_idx + chunk_size, len(df))
                remaining = total_rows - rows_processed
                
                if remaining <= 0:
                    break
                    
                # Slice the dataframe to respect both chunk_size and max_rows
                slice_size = min(end_idx - start_idx, remaining)
                chunk_df = df.iloc[start_idx:start_idx + slice_size]
                
                chunks.append(chunk_df)
                rows_processed += len(chunk_df)
                start_idx += chunk_size
                
                if rows_processed >= total_rows:
                    break
        
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def read_table_with_row_cap(path, max_rows, **read_kwargs):
    """Read parquet table up to max_rows using row group optimization"""
    parquet_file = pq.ParquetFile(path)
    total_rows = 0
    row_groups = []

    for rg_idx in range(parquet_file.num_row_groups):
        rg_rows = parquet_file.metadata.row_group(rg_idx).num_rows
        if total_rows + rg_rows > max_rows:
            break
        row_groups.append(rg_idx)
        total_rows += rg_rows

    # Read full row groups
    if not row_groups:
        # Handle case where max_rows is less than first row group size
        partial_table = parquet_file.read_row_group(0, **read_kwargs).slice(0, max_rows)
        return partial_table
    
    table = parquet_file.read_row_groups(row_groups, **read_kwargs)
    
    # Read partial last row group if needed
    if total_rows < max_rows and len(row_groups) < parquet_file.num_row_groups:
        partial_rows = max_rows - total_rows
        last_rg_idx = len(row_groups)
        try:
            # Try to read with row_groups API (newer PyArrow)
            partial_table = parquet_file.read_row_groups(
                [last_rg_idx], 
                **read_kwargs
            ).slice(0, partial_rows)
            table = pa.concat_tables([table, partial_table])
        except (AttributeError, NotImplementedError):
            # Fallback for older PyArrow versions
            partial_table = parquet_file.read_row_group(
                last_rg_idx,
                **read_kwargs
            ).slice(0, partial_rows)
            table = pa.concat_tables([table, partial_table])
    
    return table


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