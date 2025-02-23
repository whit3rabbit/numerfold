import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy.stats import pearsonr
import torch

def evaluate_numerai_metrics(predictions_df: pd.DataFrame,
                             targets: Union[str, List[str]],
                             era_col: str = 'era') -> Dict[str, float]:
    """
    Calculate Numerai-specific metrics with improved error handling and validation.

    This function computes evaluation metrics such as mean correlation, standard deviation
    of correlations, Sharpe ratio, and overall correlation. It operates era-wise to capture
    temporal variations and validates the presence of necessary columns.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions and target values.
        targets (Union[str, List[str]]): Target column name(s); if multiple are provided, the first is used.
        era_col (str, optional): Column name representing eras. Defaults to 'era'.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics:
            - mean_correlation: Mean of era-wise correlations.
            - std_correlation: Standard deviation of era-wise correlations.
            - sharpe_ratio: Ratio of mean correlation to its standard deviation.
            - overall_correlation: Correlation computed over all data.
            - feature_neutral_correlation: Simplified as overall correlation.
            - worst_era_correlation: Minimum era-wise correlation.
            - best_era_correlation: Maximum era-wise correlation.
            - num_eras: Number of eras with valid correlations.
    """
    try:
        # Ensure targets is a list.
        if isinstance(targets, str):
            targets = [targets]

        # Required columns.
        required_cols = ['prediction', era_col] + targets
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        target_col = targets[0]

        # Convert prediction and target columns to numeric.
        predictions_df['prediction'] = pd.to_numeric(predictions_df['prediction'], errors='coerce')
        predictions_df[target_col] = pd.to_numeric(predictions_df[target_col], errors='coerce')

        # Filter valid rows.
        valid_mask = ~(predictions_df['prediction'].isna() | predictions_df[target_col].isna())
        if not valid_mask.any():
            raise ValueError("No valid prediction/target pairs found")
        predictions_df = predictions_df[valid_mask].copy()

        # Compute era-wise correlations.
        era_correlations = {}
        all_eras = predictions_df[era_col].unique()
        for era in all_eras:
            era_data = predictions_df[predictions_df[era_col] == era]
            if len(era_data) > 1:
                try:
                    corr = pearsonr(era_data['prediction'], era_data[target_col])[0]
                    if not np.isnan(corr):
                        era_correlations[era] = corr
                except Exception:
                    continue

        if not era_correlations:
            raise ValueError("No valid correlations calculated")

        correlations = np.array(list(era_correlations.values()))
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations) if len(correlations) > 1 else 1e-10
        sharpe_ratio = mean_correlation / std_correlation if std_correlation > 0 else 0

        overall_correlation = pearsonr(
            predictions_df['prediction'],
            predictions_df[target_col]
        )[0]

        return {
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'sharpe_ratio': float(sharpe_ratio),
            'overall_correlation': float(overall_correlation),
            'feature_neutral_correlation': float(overall_correlation),
            'worst_era_correlation': float(np.min(correlations)),
            'best_era_correlation': float(np.max(correlations)),
            'num_eras': len(era_correlations)
        }

    except Exception as e:
        print(f"Error in metric evaluation: {str(e)}")
        return {
            'mean_correlation': 0.0,
            'std_correlation': 1e-10,
            'sharpe_ratio': 0.0,
            'overall_correlation': 0.0,
            'feature_neutral_correlation': 0.0,
            'worst_era_correlation': 0.0,
            'best_era_correlation': 0.0,
            'num_eras': 0
        }


def generate_model_predictions(model: torch.nn.Module,
                               data_loader: torch.utils.data.DataLoader,
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and confidence scores from the model.

    This function iterates over the data_loader, moves data to the specified device,
    obtains predictions from the model, and computes confidence scores. If the model has
    a 'forward_dict' method, it uses that to extract predictions and confidence; otherwise,
    it defaults to using the standard forward method.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader containing input data.
        device (str, optional): Device to run the model on. Defaults to 'cuda' if available.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - predictions: Array of model predictions.
            - confidences: Array of corresponding confidence scores.
    """
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            try:
                data = data.to(device)

                if hasattr(model, 'forward_dict'):
                    outputs = model.forward_dict(data)
                    batch_preds = outputs['prediction']
                    batch_conf = outputs.get('confidence', torch.ones_like(batch_preds))
                else:
                    batch_preds, _ = model(data)
                    batch_conf = torch.ones_like(batch_preds)

                batch_preds = batch_preds.view(-1)
                batch_conf = batch_conf.view(-1)

                predictions.extend(batch_preds.cpu().numpy())
                confidences.extend(batch_conf.cpu().numpy())

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                batch_size = data.shape[0]
                predictions.extend([0.5] * batch_size)
                confidences.extend([0.0] * batch_size)

    return np.array(predictions), np.array(confidences)


def print_evaluation_results(results: Dict) -> None:
    """
    Print evaluation results in a formatted way.

    Args:
        results (Dict): Dictionary containing evaluation results.
    """
    print("\nEvaluation Results:")
    print("-" * 50)

    if results.get('error'):
        print(f"Error during evaluation: {results['error']}")
        return

    print("Standard Predictions:")
    if results.get('standard_metrics'):
        for metric, value in results['standard_metrics'].items():
            print(f"  {metric}: {value:.6f}")

    print("\nConfidence-Weighted Predictions:")
    if results.get('weighted_metrics'):
        for metric, value in results['weighted_metrics'].items():
            print(f"  {metric}: {value:.6f}")

    print("\nConfidence Statistics:")
    print(f"  Mean Confidence: {results.get('mean_confidence', 0):.6f}")
    print(f"  Number of Predictions: {results.get('num_predictions', 0)}")


def run_final_evaluation(val_df: pd.DataFrame,
                         model: torch.nn.Module,
                         val_loader: torch.utils.data.DataLoader,
                         targets: Union[str, List[str]],
                         era_col: str = 'era',
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """
    Run final model evaluation using the trained transformer model.

    Args:
        val_df (pd.DataFrame): Validation DataFrame.
        model (torch.nn.Module): Trained model.
        val_loader (torch.utils.data.DataLoader): Validation DataLoader.
        targets (Union[str, List[str]]): Target column name(s).
        era_col (str): Era column name.
        device (str): Device to run model on.

    Returns:
        Dict: Dictionary containing evaluation results, including:
            - val_df: DataFrame with predictions and confidence scores.
            - standard_metrics: Metrics for standard predictions.
            - weighted_metrics: Metrics for confidence-weighted predictions.
            - mean_confidence: Mean confidence score.
            - num_predictions: Number of predictions made.
    """
    try:
        # Generate model predictions.
        predictions, confidences = generate_model_predictions(model, val_loader, device)

        # Create evaluation DataFrame.
        eval_df = val_df.copy()
        eval_df['prediction'] = predictions
        eval_df['confidence'] = confidences
        eval_df['confidence_weighted_prediction'] = predictions * confidences

        # Calculate metrics for standard predictions.
        standard_metrics = evaluate_numerai_metrics(eval_df, targets, era_col)

        # Calculate metrics for confidence-weighted predictions.
        eval_df['prediction'] = eval_df['confidence_weighted_prediction']
        weighted_metrics = evaluate_numerai_metrics(eval_df, targets, era_col)

        return {
            'val_df': eval_df,
            'standard_metrics': standard_metrics,
            'weighted_metrics': weighted_metrics,
            'mean_confidence': float(np.mean(confidences)),
            'num_predictions': len(predictions)
        }

    except Exception as e:
        print(f"Error in final evaluation: {str(e)}")
        return {
            'error': str(e),
            'val_df': val_df,
            'standard_metrics': None,
            'weighted_metrics': None,
            'mean_confidence': 0.0,
            'num_predictions': 0
        }
