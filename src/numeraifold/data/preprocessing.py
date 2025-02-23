import numpy as np
import pandas as pd
import torch

def create_input_tensor(df: pd.DataFrame, feature_list: list, batch_indices=None) -> torch.Tensor:
    """
    Create a PyTorch tensor from a DataFrame using the specified feature list.

    This function converts the DataFrame into a tensor with shape [num_samples, num_features].
    If batch_indices are provided, only the corresponding rows are processed.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing feature data.
        feature_list (list): List of column names to be used as features.
        batch_indices (list, optional): List of row indices to process. Defaults to None.

    Returns:
        torch.Tensor: A tensor of type float32 with shape [num_samples, num_features].
                      Returns None if an error occurs during conversion.
    """
    try:
        # Use only the specified rows if batch_indices is provided, else use the full DataFrame.
        df_subset = df.loc[batch_indices] if batch_indices is not None else df

        # Determine the shape of the output array.
        num_samples = len(df_subset)
        num_features = len(feature_list)
        feature_array = np.zeros((num_samples, num_features), dtype=np.float32)

        # Iterate over each feature and fill the corresponding column in the array.
        for i, feature in enumerate(feature_list):
            try:
                # Convert feature values to numeric, handling errors and missing values.
                values = pd.to_numeric(df_subset[feature], errors='coerce').fillna(0).values
                feature_array[:, i] = values
            except Exception as e:
                print(f"Warning: Error converting feature {feature}: {e}")
                # If conversion fails, the column remains as zeros.

        # Convert the numpy array into a PyTorch tensor.
        tensor = torch.tensor(feature_array, dtype=torch.float32)
        return tensor

    except Exception as e:
        print(f"Error creating input tensor: {e}")
        return None


def debug_tensor_shape(tensor: torch.Tensor, name: str, detailed: bool = False) -> None:
    """
    Print debug information for a given tensor.

    Outputs the shape, device, dtype, and optionally detailed statistics (min, max, mean, NaN/Inf checks).

    Parameters:
        tensor (torch.Tensor): The tensor to be debugged.
        name (str): A label for the tensor.
        detailed (bool, optional): If True, print detailed statistics. Default is False.
    """
    print(f"\n=== {name} Debug Info ===")
    print(f"Shape: {tensor.shape}")
    if isinstance(tensor, torch.Tensor):
        print(f"Device: {tensor.device}")
        print(f"Dtype: {tensor.dtype}")
        if detailed:
            print(f"Min value: {tensor.min().item():.3f}")
            print(f"Max value: {tensor.max().item():.3f}")
            print(f"Mean value: {tensor.mean().item():.3f}")
            print(f"Any NaN: {torch.isnan(tensor).any().item()}")
            print(f"Any Inf: {torch.isinf(tensor).any().item()}")