"""
Core package for NumerAIFold and related utilities.

This package includes:
  - Models: NumerAIFold, TransformerBlock, MultiHeadSelfAttention, FeedForward, ImprovedPairwiseAttention.
  - Training utilities: train_numeraifold_model, create_data_batches, calculate_confidence_scores.
  - Evaluation functions: evaluate_numerai_metrics, generate_model_predictions, print_evaluation_results, run_final_evaluation.
"""

# Import models from the model module.
from .model import (
    NumerAIFold,
    TransformerBlock,
    MultiHeadSelfAttention,
    FeedForward,
    ImprovedPairwiseAttention
)

# Import training functions from the training module.
from .training import (
    train_numeraifold_model,
    create_data_batches,
    calculate_confidence_scores
)

# Import evaluation functions from the evaluation module.
from .evaluation import (
    evaluate_numerai_metrics,
    generate_model_predictions,
    print_evaluation_results,
    run_final_evaluation
)

# Define the public API for the package.
__all__ = [
    "NumerAIFold",
    "TransformerBlock",
    "MultiHeadSelfAttention",
    "FeedForward",
    "ImprovedPairwiseAttention",
    "train_numeraifold_model",
    "create_data_batches",
    "calculate_confidence_scores",
    "evaluate_numerai_metrics",
    "generate_model_predictions",
    "print_evaluation_results",
    "run_final_evaluation"
]

__version__ = "0.1.0"
