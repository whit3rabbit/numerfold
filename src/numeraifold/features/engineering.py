# Standard library and third-party imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from numerblox.preprocessing import GroupStatsPreProcessor

from numeraifold.data.loading import process_in_batches
from numeraifold.core.model import NumerAIFold

class AlphaFoldFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline inspired by AlphaFold's approach to protein domains.

    This transformer performs the following:
    1. Identifies "domains" within financial features (groups of related features)
    2. Computes evolutionary-inspired statistics within and across domains
    3. Generates embeddings that capture both local and global relationships
    """
    def __init__(self, era_col='era', feature_groups=None, embedding_size=64):
        """
        Initialize the AlphaFoldFeatureEngineering transformer.

        Parameters:
            era_col (str): Column name in the DataFrame that indicates eras.
            feature_groups (dict): A dictionary mapping domain names to lists of feature indices or names.
            embedding_size (int): Size of the domain embedding vector.
        """
        self.era_col = era_col
        # Default feature groups if none provided
        self.feature_groups = feature_groups or {
            'intelligence': [f for f in range(1, 21)],
            'value': [f for f in range(21, 41)],
            'momentum': [f for f in range(41, 61)],
            'size': [f for f in range(61, 81)],
            'volatility': [f for f in range(81, 101)]
        }
        self.embedding_size = embedding_size
        self.group_processors = {}   # Will store GroupStatsPreProcessor objects per domain
        self.domain_embeddings = {}  # Will store domain embeddings generated during fit

    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline on the training data.

        Parameters:
            X (pd.DataFrame): Training features.
            y (pd.Series or np.array, optional): Target variable used for training embeddings.

        Returns:
            self: Fitted transformer.
        """
        # Create and fit group statistic processors for each feature group
        for group_name, feature_indices in self.feature_groups.items():
            # Convert indices to column names if necessary
            if isinstance(feature_indices[0], int):
                feature_names = [f'feature_{i}' for i in feature_indices]
            else:
                feature_names = feature_indices

            # Initialize and fit the group processor for current feature group
            processor = GroupStatsPreProcessor(groups=[group_name])
            processor.fit(X[feature_names])
            self.group_processors[group_name] = processor

        # Train domain embeddings using LightGBM for dimension reduction
        for group_name, processor in self.group_processors.items():
            # Transform features using the group processor
            group_features = processor.transform(X)

            # Create a LightGBM model to generate embeddings from domain features
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=3,
                num_leaves=31,
                colsample_bytree=0.8
            )

            # If target is provided, fit the model to encode domain knowledge
            if y is not None:
                model.fit(group_features, y)

                # Use feature importances as an embedding representation for the domain
                importances = model.feature_importances_
                self.domain_embeddings[group_name] = importances

        return self

    def transform(self, X):
        """
        Transform the input data by adding domain-specific and cross-domain features.

        Parameters:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with additional engineered features.
        """
        result_df = X.copy()

        # Generate domain-specific features using the fitted processors
        for group_name, processor in self.group_processors.items():
            # Obtain processed domain features
            domain_features = processor.transform(X)
            # Add a prefix to each column to indicate its domain
            domain_features = domain_features.add_prefix(f'domain_{group_name}_')
            # Merge the new features with the original DataFrame
            result_df = pd.concat([result_df, domain_features], axis=1)

        # Create features that capture interactions between different domains
        result_df = self._create_cross_domain_features(result_df)
        # Add era-wise statistical features inspired by evolutionary analysis
        result_df = self._add_era_statistics(result_df)

        return result_df

    def _create_cross_domain_features(self, df):
        """
        Create features that capture relationships between domains.

        Parameters:
            df (pd.DataFrame): DataFrame with domain-specific features.

        Returns:
            pd.DataFrame: DataFrame with added cross-domain features.
        """
        result_df = df.copy()

        # Gather domain-specific column names for each group
        domain_cols = {}
        for group_name in self.group_processors.keys():
            domain_cols[group_name] = [col for col in df.columns if f'domain_{group_name}_' in col]

        # Generate cross-domain features for each unique pair of domains
        for i, (domain1, cols1) in enumerate(domain_cols.items()):
            for domain2, cols2 in list(domain_cols.items())[i+1:]:
                # Skip if one of the domains has no features
                if not cols1 or not cols2:
                    continue

                # Create ratio features based on key statistical measures
                for stat in ['mean', 'std', 'min', 'max']:
                    col1 = next((c for c in cols1 if f'_{stat}' in c), None)
                    col2 = next((c for c in cols2 if f'_{stat}' in c), None)
                    if col1 and col2:
                        ratio_name = f'xdomain_{domain1}_{domain2}_{stat}_ratio'
                        # Avoid division by zero by adding a small constant
                        result_df[ratio_name] = df[col1] / (df[col2] + 1e-8)

                # Create correlation features between domain columns
                corr_name = f'xdomain_{domain1}_{domain2}_corr'
                result_df[corr_name] = df[cols1].corrwith(df[cols2], axis=1).fillna(0)

        return result_df

    def _add_era_statistics(self, df):
        """
        Add era-level statistical features inspired by AlphaFold's evolutionary analysis.

        Parameters:
            df (pd.DataFrame): DataFrame to which era statistics will be added.

        Returns:
            pd.DataFrame: DataFrame with era-wise z-scores, ranks, and median distances.
        """
        result_df = df.copy()

        if self.era_col in df.columns:
            # Select numeric columns for era-based computations
            numeric_cols = df.select_dtypes(include=np.number).columns

            for col in numeric_cols:
                # Skip the era identifier column itself
                if col == self.era_col:
                    continue

                # Compute era-wise mean and standard deviation
                era_mean = df.groupby(self.era_col)[col].transform('mean')
                era_std = df.groupby(self.era_col)[col].transform('std').fillna(1)  # Avoid division by zero

                # Calculate the z-score for the column within each era
                z_score_col = f'{col}_era_z'
                result_df[z_score_col] = (df[col] - era_mean) / era_std

                # Rank values within each era (percentile rank)
                rank_col = f'{col}_era_rank'
                result_df[rank_col] = df.groupby(self.era_col)[col].rank(pct=True)

                # Compute absolute distance from the era median
                median_col = f'{col}_era_median_dist'
                era_median = df.groupby(self.era_col)[col].transform('median')
                result_df[median_col] = (df[col] - era_median).abs()

        return result_df


def generate_alphafold_features(train_df, val_df, model, features, confidence_threshold=0.5, device='cuda'):
    """
    Generate AlphaFold-inspired features from training and validation data.

    This function processes the data in batches, generates embeddings, predictions,
    and confidence scores, and then constructs new DataFrames with these features.

    Parameters:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        model: Model used for feature extraction.
        features (list): List of feature column names to use.
        confidence_threshold (float): Threshold to mark high confidence predictions.
        device (str): Device to use for model inference (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (train_df_new, val_df_new) DataFrames with engineered features,
               or (None, None) if an error occurs.
    """
    print("Generating AlphaFold-inspired features...")

    def process_predictions(preds):
        """
        Ensure predictions are one-dimensional.

        Parameters:
            preds (np.array): Array of predictions.

        Returns:
            np.array: Flattened prediction array.
        """
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                return preds.squeeze()
            else:
                print(f"Multiple prediction columns found, using first column. Shape: {preds.shape}")
                return preds[:, 0]
        return preds

    # Process training data in batches
    print("Processing training data...")
    train_embeddings, train_confidences, train_preds = process_in_batches(
        train_df, model, features, device
    )

    if train_embeddings is None:
        print("Failed to generate training features")
        return None, None

    print(f"Training data shapes - embeddings: {train_embeddings.shape}, "
          f"confidences: {train_confidences.shape}, predictions: {train_preds.shape}")

    # Process validation data in batches
    print("Processing validation data...")
    val_embeddings, val_confidences, val_preds = process_in_batches(
        val_df, model, features, device
    )

    if val_embeddings is None:
        print("Failed to generate validation features")
        return None, None

    print(f"Validation data shapes - embeddings: {val_embeddings.shape}, "
          f"confidences: {val_confidences.shape}, predictions: {val_preds.shape}")

    try:
        # Ensure predictions are 1D arrays
        train_preds_flat = process_predictions(train_preds)
        val_preds_flat = process_predictions(val_preds)

        print(f"Processed prediction shapes - train: {train_preds_flat.shape}, "
              f"val: {val_preds_flat.shape}")

        # Verify that the dimensions of data, predictions, and confidences match
        if not (len(train_df.index) == len(train_preds_flat) == len(train_confidences)):
            raise ValueError(f"Mismatched dimensions in training data: "
                             f"index={len(train_df.index)}, predictions={len(train_preds_flat)}, "
                             f"confidences={len(train_confidences)}")

        if not (len(val_df.index) == len(val_preds_flat) == len(val_confidences)):
            raise ValueError(f"Mismatched dimensions in validation data: "
                             f"index={len(val_df.index)}, predictions={len(val_preds_flat)}, "
                             f"confidences={len(val_confidences)}")

        # Create column names for the embeddings
        train_embedding_columns = [f'af_emb_{i}' for i in range(train_embeddings.shape[1])]
        val_embedding_columns = [f'af_emb_{i}' for i in range(val_embeddings.shape[1])]

        # Construct new DataFrame for training data with embeddings and additional features
        train_df_new = pd.DataFrame(
            train_embeddings,
            index=train_df.index,
            columns=train_embedding_columns
        )
        train_df_new['af_prediction'] = train_preds_flat  # Original prediction from model
        train_df_new['prediction'] = train_preds_flat       # Required column for evaluation
        train_df_new['af_confidence'] = train_confidences
        train_df_new['af_high_confidence'] = (train_confidences > confidence_threshold).astype(int)

        # Construct new DataFrame for validation data with embeddings and additional features
        val_df_new = pd.DataFrame(
            val_embeddings,
            index=val_df.index,
            columns=val_embedding_columns
        )
        val_df_new['af_prediction'] = val_preds_flat       # Original prediction from model
        val_df_new['prediction'] = val_preds_flat            # Required column for evaluation
        val_df_new['af_confidence'] = val_confidences
        val_df_new['af_high_confidence'] = (val_confidences > confidence_threshold).astype(int)

        # Final shape verification
        print(f"Final DataFrame shapes - train: {train_df_new.shape}, val: {val_df_new.shape}")

        # Check that all required columns are present
        required_cols = ['prediction', 'af_prediction', 'af_confidence', 'af_high_confidence']
        missing_train = [col for col in required_cols if col not in train_df_new.columns]
        missing_val = [col for col in required_cols if col not in val_df_new.columns]

        if missing_train or missing_val:
            raise ValueError(f"Missing required columns - train: {missing_train}, val: {missing_val}")

        # Create a new column for confidence-weighted predictions
        train_df_new['confidence_weighted_prediction'] = train_df_new['prediction'] * train_df_new['af_confidence']
        val_df_new['confidence_weighted_prediction'] = val_df_new['prediction'] * val_df_new['af_confidence']

        return train_df_new, val_df_new

    except Exception as e:
        print(f"Error creating feature DataFrames: {str(e)}")
        print("Debug information:")
        print(f"Train embeddings type: {type(train_embeddings)}, "
              f"shape: {train_embeddings.shape if hasattr(train_embeddings, 'shape') else 'unknown'}")
        print(f"Train confidences type: {type(train_confidences)}, "
              f"shape: {train_confidences.shape if hasattr(train_confidences, 'shape') else 'unknown'}")
        print(f"Train predictions type: {type(train_preds)}, "
              f"shape: {train_preds.shape if hasattr(train_preds, 'shape') else 'unknown'}")
        return None, None


def create_alphafold_features(df, feature_cols, model_path, output_prefix='af_'):
    """
    Generate AlphaFold-inspired features for Numerai data.

    This function uses a feature extractor to create embeddings and then appends them
    as new feature columns to the original DataFrame.

    Parameters:
        df (pd.DataFrame): Input data.
        feature_cols (list): List of columns to use for feature extraction.
        model_path (str): Path to the pre-trained model.
        output_prefix (str): Prefix for naming the new embedding columns.

    Returns:
        pd.DataFrame: Original DataFrame concatenated with the new embedding features.
    """
    # Initialize the feature extractor with the given model path
    extractor = NumerAIFoldFeatureExtractor(model_path)
    # Extract embeddings for the provided features
    embeddings = extractor.extract_features(df, feature_cols)

    # Create a new DataFrame with embedding features and proper column names
    embedding_df = pd.DataFrame(
        embeddings,
        index=df.index,
        columns=[f"{output_prefix}{i}" for i in range(embeddings.shape[1])]
    )

    # Concatenate the original DataFrame with the new embedding features
    return pd.concat([df, embedding_df], axis=1)

class NumerAIFoldFeatureExtractor:
    """Extract features using a pre-trained NumerAIFold model."""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = NumerAIFold(num_features=None)  # Will be set during loading
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)

    def extract_features(self, df, feature_cols, batch_size=64):
        """Extract embeddings from the model's penultimate layer."""
        dataset = self._prepare_dataset(df, feature_cols)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                # Forward pass through all but the last layer
                x = self.model.feature_embedding(batch)
                x = x + self.model.pos_embedding[:, :batch.size(1), :]

                for transformer in self.model.transformer_blocks:
                    x, _ = transformer(x)

                x = self.model.output_norm(x)
                # Use mean pooling across sequence dimension
                pooled_features = x.mean(dim=1)
                embeddings.append(pooled_features.cpu().numpy())

        return np.vstack(embeddings)

    def _prepare_dataset(self, df, feature_cols):
        features = df[feature_cols].values
        # Reshape to [n_samples, n_features, 1] so each feature becomes a "residue"
        features = features.reshape(features.shape[0], features.shape[1], 1)
        return torch.tensor(features, dtype=torch.float32)
