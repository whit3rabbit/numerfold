"""
__init__.py

This package provides data loading utilities including functions for:
- Loading and processing data (loading.py)
- Creating PyTorch DataLoaders and era-aware batch generators (dataloader.py)
- Preprocessing utilities such as creating input tensors and debugging tensor shapes (preprocessing.py)
"""

from .loading import load_data, process_in_batches, create_model_inputs
from .dataloader import get_dataloaders, create_data_batches
from .preprocessing import create_input_tensor, debug_tensor_shape
