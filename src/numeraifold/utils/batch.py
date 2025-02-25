import numpy as np
import pandas as pd
import torch
import gc
import psutil
import pyarrow.parquet as pq
from tqdm import tqdm

def safe_float32_conversion(series):
    """
    Safely convert a pandas Series to float32.
    
    Args:
        series: pandas Series to convert
    
    Returns:
        Converted pandas Series
    """
    try:
        # Handle non-numeric values
        if series.dtype == 'object':
            # Try to convert to numeric first
            series = pd.to_numeric(series, errors='coerce')
        
        # Convert to float32
        return series.astype(np.float32)
    except Exception as e:
        print(f"Error converting {series.name} to float32: {e}")
        # Return original if conversion fails
        return series

def batch_update_dataframe(df, columns, dtype=np.float32, batch_size=1000):
    """
    Update datatypes of DataFrame columns in batches to reduce memory spikes.
    
    Args:
        df: DataFrame to update
        columns: List of columns to convert
        dtype: Target datatype (default: np.float32)
        batch_size: Number of columns to process at once
    
    Returns:
        Updated DataFrame
    """
    # Process in batches to avoid memory spikes
    for i in range(0, len(columns), batch_size):
        batch_columns = columns[i:i+batch_size]
        print(f"Converting batch {i//batch_size + 1}/{(len(columns) + batch_size - 1)//batch_size}")
        
        for col in batch_columns:
            if col in df.columns:
                df[col] = safe_float32_conversion(df[col])
        
        # Free memory
        gc.collect()
    
    return df

def efficient_parquet_loading(file_path, feature_list=None, target_list=None, chunks=None):
    """
    Load a parquet file efficiently with proper type handling.
    
    Args:
        file_path: Path to the parquet file
        feature_list: List of feature columns to load
        target_list: List of target columns to load
        chunks: Number of chunks to split the loading into (None for auto)
    
    Returns:
        DataFrame with optimized dtypes
    """
    # Get file information
    parquet_file = pq.ParquetFile(file_path)
    num_rows = parquet_file.metadata.num_rows
    
    # Determine columns to load
    if feature_list is None and target_list is None:
        # Load all columns
        columns = None
    else:
        columns = []
        if feature_list is not None:
            columns.extend(feature_list)
        if target_list is not None:
            if isinstance(target_list, list):
                columns.extend(target_list)
            else:
                columns.append(target_list)
    
    # Determine chunk size
    if chunks is None:
        # Auto-determine chunk size based on available memory
        available_memory = psutil.virtual_memory().available
        estimated_row_size = 1000  # Conservative estimate in bytes per row
        chunk_size = max(1000, int(available_memory * 0.3 / estimated_row_size))
        chunk_size = min(chunk_size, num_rows)  # Don't exceed total rows
    else:
        chunk_size = max(1, num_rows // chunks)
    
    print(f"Loading {file_path} with {num_rows} rows in chunks of {chunk_size}")
    
    # Load in chunks
    result_chunks = []
    for i in range(0, num_rows, chunk_size):
        # Read chunk
        chunk = pd.read_parquet(
            file_path,
            columns=columns,
            skip_rows=i,
            num_rows=min(chunk_size, num_rows - i)
        )
        
        # Convert features to float32
        if feature_list is not None:
            valid_features = [f for f in feature_list if f in chunk.columns]
            for col in valid_features:
                chunk[col] = safe_float32_conversion(chunk[col])
        
        # Convert targets to float32
        if target_list is not None:
            if isinstance(target_list, list):
                valid_targets = [t for t in target_list if t in chunk.columns]
                for col in valid_targets:
                    chunk[col] = safe_float32_conversion(chunk[col])
            elif target_list in chunk.columns:
                chunk[target_list] = safe_float32_conversion(chunk[target_list])
        
        result_chunks.append(chunk)
        
        # Report progress
        print(f"Loaded chunk {i//chunk_size + 1}/{(num_rows + chunk_size - 1)//chunk_size}, "
              f"memory usage: {chunk.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    
    # Combine chunks
    result = pd.concat(result_chunks, axis=0)
    
    # Clean up to free memory
    del result_chunks
    gc.collect()
    
    return result

def batch_process_features(model, dataframe, features, batch_size=1000, device=None):
    """
    Process features through a model in batches to avoid memory issues.
    
    Args:
        model: PyTorch model
        dataframe: DataFrame with features
        features: List of feature columns
        batch_size: Size of batches to process
        device: PyTorch device (None for auto)
    
    Returns:
        Dict with model outputs (embeddings, predictions, confidences)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Prepare to collect outputs
    all_embeddings = []
    all_predictions = []
    all_confidences = []
    
    # Process in batches
    num_samples = len(dataframe)
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        # Get batch of data
        batch_df = dataframe.iloc[i:i+batch_size]
        
        # Ensure all features exist
        valid_features = [f for f in features if f in batch_df.columns]
        if len(valid_features) < len(features):
            print(f"Warning: {len(features) - len(valid_features)} features missing in batch")
        
        # Convert to float32 numpy array
        X = np.array(batch_df[valid_features].fillna(0).values, dtype=np.float32)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Process with model
        with torch.no_grad():
            outputs = model(X_tensor)
            batch_embeddings = outputs['embeddings'].cpu().numpy()
            batch_predictions = outputs['predictions'].cpu().numpy()
            batch_confidences = outputs['confidence'].cpu().numpy()
        
        # Collect outputs
        all_embeddings.append(batch_embeddings)
        all_predictions.append(batch_predictions)
        all_confidences.append(batch_confidences)
        
        # Free memory
        del X, X_tensor, batch_embeddings, batch_predictions, batch_confidences
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Combine all outputs
    embeddings = np.vstack(all_embeddings)
    predictions = np.vstack(all_predictions)
    confidences = np.vstack(all_confidences)
    
    return {
        'embeddings': embeddings,
        'predictions': predictions,
        'confidences': confidences
    }

def efficient_correlation_calculation(train_df, features, target, chunk_size=10000):
    """
    Calculate correlation between features and target efficiently for large datasets.
    
    Args:
        train_df: Training DataFrame
        features: List of feature columns
        target: Target column
        chunk_size: Size of chunks to process
    
    Returns:
        Series with correlations
    """
    # For smaller datasets, use pandas directly
    if len(train_df) <= chunk_size:
        return train_df[features].corrwith(train_df[target]).abs()
    
    # For larger datasets, calculate in chunks
    print(f"Calculating correlations in chunks for {len(features)} features...")
    
    # Initialize correlation sums and counts
    correlation_sums = pd.Series(0.0, index=features)
    chunk_count = 0
    
    # Process in chunks
    for i in range(0, len(train_df), chunk_size):
        chunk = train_df.iloc[i:i+chunk_size]
        
        # Calculate correlations for this chunk
        chunk_correlations = chunk[features].corrwith(chunk[target]).abs()
        
        # Add to running totals
        correlation_sums += chunk_correlations
        chunk_count += 1
    
    # Calculate average correlations
    average_correlations = correlation_sums / chunk_count
    
    return average_correlations

def create_sparse_representation(df, features, threshold=0.001):
    """
    Create a sparse representation of features to save memory.
    Values close to zero (below threshold) are set to exactly zero.
    
    Args:
        df: DataFrame with features
        features: List of feature columns
        threshold: Threshold for considering a value as zero
    
    Returns:
        DataFrame with sparse representation of features
    """
    from scipy import sparse
    import pandas as pd
    
    result_df = df.copy()
    
    # Convert features to sparse representation
    for col in features:
        if col in result_df.columns:
            # Get column values as numpy array
            col_values = result_df[col].values
            
            # Set values below threshold to zero
            col_values[np.abs(col_values) < threshold] = 0
            
            # Create sparse array
            sparse_array = sparse.csr_matrix(col_values.reshape(-1, 1))
            
            # Replace column with sparse array values
            result_df[col] = pd.Series(sparse_array.toarray().flatten())
    
    return result_df