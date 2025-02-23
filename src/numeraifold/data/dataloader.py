"""
dataloader.py

This module provides functions to create PyTorch DataLoaders and data batches
for training and validation. It handles robust type conversion, error reporting,
and memory-efficient processing.
"""

import pandas as pd
import numpy as np
import torch


def get_dataloaders(train_df, val_df, features, targets, batch_size=64):
    """
    Create PyTorch DataLoaders for training and validation with robust type conversion.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame containing features and targets.
        val_df (pd.DataFrame): Validation DataFrame containing features and targets.
        features (list): List of feature column names.
        targets (list or str or iterable): Target column names. If a string, it will be converted to a list.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.

    Returns:
        tuple: A tuple (train_loader, val_loader) containing the training and validation DataLoaders.
    """
    print("Creating data loaders with type checking...")

    # Convert targets to a list if necessary
    if isinstance(targets, str):
        targets = [targets]
    elif not isinstance(targets, list):
        targets = list(targets)

    # Ensure there is at least one target
    if not targets:
        targets = ['target']

    # Verify that features and targets exist in both DataFrames
    valid_features = [f for f in features if f in train_df.columns and f in val_df.columns]
    if len(valid_features) < len(features):
        missing = set(features) - set(valid_features)
        print(f"Warning: {len(missing)} features not found in both datasets: {missing}")
        features = valid_features

    valid_targets = [t for t in targets if t in train_df.columns and t in val_df.columns]
    if len(valid_targets) < len(targets):
        missing = set(targets) - set(valid_targets)
        print(f"Warning: {len(missing)} targets not found in both datasets: {missing}")
        targets = valid_targets

    if not features:
        raise ValueError("No valid features found in both datasets")
    if not targets:
        raise ValueError("No valid targets found in both datasets")

    try:
        # Check data types and convert columns if needed
        for df in [train_df, val_df]:
            for col in features + targets:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"Warning: Converting non-numeric column '{col}' to float")
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    except Exception as e:
                        print(f"Error converting column '{col}': {e}")
                        raise ValueError(f"Column '{col}' could not be converted to numeric")

        # Process DataFrame to tensor in chunks if needed
        def process_df_to_tensor(df, columns, dtype=torch.float32, reshape_3d=False):
            """Convert DataFrame columns to a tensor, processing in chunks if the dataset is large."""
            if len(df) > 100000:
                chunk_size = 50000
                tensors = []
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk_array = chunk[columns].astype(np.float32).fillna(0).values
                    if reshape_3d and len(chunk_array.shape) == 2:
                        chunk_array = chunk_array.reshape(chunk_array.shape[0], chunk_array.shape[1], 1)
                    tensors.append(torch.tensor(chunk_array, dtype=dtype))
                return torch.cat(tensors, dim=0)
            else:
                array_data = df[columns].astype(np.float32).fillna(0).values
                if reshape_3d and len(array_data.shape) == 2:
                    array_data = array_data.reshape(array_data.shape[0], array_data.shape[1], 1)
                return torch.tensor(array_data, dtype=dtype)

        # Create feature and target tensors for training and validation
        X_train = process_df_to_tensor(train_df, features, reshape_3d=True)
        y_train = process_df_to_tensor(train_df, targets)
        X_val = process_df_to_tensor(val_df, features, reshape_3d=True)
        y_val = process_df_to_tensor(val_df, targets)

        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        # Create DataLoaders with efficient settings
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0,  # Adjust based on your system
            drop_last=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0,
            drop_last=False
        )

        print(f"Successfully created dataloaders with {len(features)} features and {len(targets)} targets")
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        print(f"Feature tensor shapes - Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"Target tensor shapes - Train: {y_train.shape}, Val: {y_val.shape}")
        return train_loader, val_loader

    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        error_details = str(e)
        print(f"Attempting simpler approach based on error: {error_details}")

        try:
            # Print feature sample for debugging
            for col in features[:3]:
                if col in train_df.columns:
                    print(f"Sample of '{col}': {train_df[col].head(3).tolist()}, type: {train_df[col].dtype}")

            # Manual conversion to numpy arrays with float type
            X_train_np = np.array(train_df[features].fillna(0).values, dtype=np.float32)
            y_train_np = np.array(train_df[targets].fillna(0).values, dtype=np.float32)
            X_val_np = np.array(val_df[features].fillna(0).values, dtype=np.float32)
            y_val_np = np.array(val_df[targets].fillna(0).values, dtype=np.float32)

            # Convert numpy arrays to tensors
            X_train = torch.tensor(X_train_np)
            y_train = torch.tensor(y_train_np)
            X_val = torch.tensor(X_val_np)
            y_val = torch.tensor(y_val_np)

            # Create TensorDatasets
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

            # Create simpler DataLoaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False)

            print("Created simplified dataloaders with manual type conversion")
            return train_loader, val_loader

        except Exception as e2:
            print(f"Both dataloader creation approaches failed: {e2}")

            if len(features) > 0 and len(targets) > 0:
                try:
                    # Use only the first feature and target as a last resort
                    feature = features[0]
                    target = targets[0]

                    X_train = torch.tensor(train_df[feature].fillna(0).astype(float).values.reshape(-1, 1))
                    y_train = torch.tensor(train_df[target].fillna(0).astype(float).values.reshape(-1, 1))
                    X_val = torch.tensor(val_df[feature].fillna(0).astype(float).values.reshape(-1, 1))
                    y_val = torch.tensor(val_df[target].fillna(0).astype(float).values.reshape(-1, 1))

                    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False)

                    print(f"WARNING: Created emergency dataloaders with only one feature ({feature}) and one target ({target})")
                    return train_loader, val_loader

                except Exception as e3:
                    print(f"All dataloader creation approaches failed: {e3}")
                    raise ValueError(f"Cannot create dataloaders. Final error: {e3}") from e3
            else:
                raise ValueError("No valid features or targets available") from e2


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
