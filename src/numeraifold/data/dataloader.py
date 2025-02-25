"""
dataloader.py

This module provides functions to create PyTorch DataLoaders and data batches
for training and validation. It handles robust type conversion, error reporting,
and memory-efficient processing.
"""

import os
import traceback
import pandas as pd
import numpy as np
import torch


def get_dataloaders(train_df, val_df, features, targets, batch_size=64):
    """
    Create PyTorch DataLoaders with proper float32 conversion.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        features: List of feature column names
        targets: List of target column names
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    print(f"Creating dataloaders with {len(features)} features and batch size {batch_size}...")
    
    # Ensure all features exist in both dataframes
    valid_features = [f for f in features if f in train_df.columns and f in val_df.columns]
    if len(valid_features) < len(features):
        print(f"Warning: {len(features) - len(valid_features)} features not found in both datasets")
        features = valid_features
    
    # Ensure target columns exist
    main_target = targets[0] if isinstance(targets, list) else targets
    
    try:
        # Handle memory-efficient conversion for large datasets
        if len(train_df) > 50000:
            print("Large dataset detected, processing in chunks...")
            
            def process_in_chunks(df, features, target_col, chunk_size=10000):
                """Process large dataframes in chunks to avoid memory issues"""
                all_X = []
                all_y = []
                
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    # Convert features to float32
                    X_chunk = np.array(chunk[features].fillna(0).values, dtype=np.float32)
                    # Convert target to float32
                    y_chunk = np.array(chunk[target_col].fillna(0).values, dtype=np.float32).reshape(-1, 1)
                    
                    all_X.append(X_chunk)
                    all_y.append(y_chunk)
                
                return np.vstack(all_X), np.vstack(all_y)
            
            # Process training data in chunks
            X_train, y_train = process_in_chunks(train_df, features, main_target)
            print(f"Processed training data: X shape {X_train.shape}, y shape {y_train.shape}")
            
            # Process validation data in chunks
            X_val, y_val = process_in_chunks(val_df, features, main_target)
            print(f"Processed validation data: X shape {X_val.shape}, y shape {y_val.shape}")
        else:
            # For smaller datasets, process all at once
            print("Processing dataset in one go...")
            # Convert features to float32
            X_train = np.array(train_df[features].fillna(0).values, dtype=np.float32)
            X_val = np.array(val_df[features].fillna(0).values, dtype=np.float32)
            
            # Convert target to float32
            y_train = np.array(train_df[main_target].fillna(0).values, dtype=np.float32).reshape(-1, 1)
            y_val = np.array(val_df[main_target].fillna(0).values, dtype=np.float32).reshape(-1, 1)
        
        # Convert to tensors with explicit dtype
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # Report memory usage
        train_tensor_mb = X_train_tensor.element_size() * X_train_tensor.nelement() / (1024 * 1024)
        val_tensor_mb = X_val_tensor.element_size() * X_val_tensor.nelement() / (1024 * 1024)
        print(f"Memory usage - Training tensors: {train_tensor_mb:.2f} MB, Validation tensors: {val_tensor_mb:.2f} MB")
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create DataLoaders with proper num_workers and pin_memory settings
        num_workers = min(4, os.cpu_count() or 1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print(traceback.format_exc())
        raise e


def create_data_batches(df, features, targets, era_col='era', batch_size=64):
    """
    Create batches for model training with era-aware sampling.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing data.
        features (list): List of feature column names.
        targets (list): List of target column names.
        era_col (str, optional): Column name representing eras. Defaults to 'era'.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        generator: A generator that yields tuples (X_batch, y_batch) for each batch.
    """
    eras = df[era_col].unique()

    # Create feature and target tensors from the DataFrame
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[targets].values, dtype=torch.float32)

    # Map each era to its row indices
    era_indices = {era: df[df[era_col] == era].index.tolist() for era in eras}

    def batch_generator():
        # Shuffle eras for randomness
        era_list = list(eras)
        np.random.shuffle(era_list)
        for era in era_list:
            indices = era_indices[era]
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                yield X[batch_indices], y[batch_indices]

    return batch_generator
