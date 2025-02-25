# Standard library and third-party imports
import traceback
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from numerblox.preprocessing import GroupStatsPreProcessor

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


def generate_alphafold_features(train_df, val_df, model, features, confidence_threshold=0.5):
    """
    Generate AlphaFold-inspired features with explicit float32 conversion.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        model: Trained NumerAIFold model
        features: List of feature column names
        confidence_threshold: Threshold for high-confidence predictions
        
    Returns:
        train_features_df, val_features_df: DataFrames with generated features
    """
    print("Generating AlphaFold-inspired features...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Process dataframes in chunks if they're large
    def process_df_in_chunks(df, chunk_size=5000):
        """Process large dataframes in chunks to avoid memory issues"""
        all_embeddings = []
        all_predictions = []
        all_confidences = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(df) + chunk_size - 1)//chunk_size}")
            
            # Convert features to float32
            X = np.array(chunk[features].fillna(0).values, dtype=np.float32)
            
            # Reshape data to [batch_size, num_features, 1] as expected by the model
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                # Get model outputs - assuming model returns (predictions, attentions)
                predictions, attentions = model(X_tensor)
                
                # Extract embeddings
                x = model.feature_embedding(X_tensor)
                x = x + model.pos_encoding[:, :X_tensor.size(1), :]
                
                for transformer in model.transformer_blocks:
                    x, _ = transformer(x)
                
                x = model.output_norm(x)
                embeddings = x.mean(dim=1).cpu().numpy()
                
                # Calculate confidence from attention patterns
                if attentions and len(attentions) > 0:
                    attention_stack = torch.stack(attentions)
                    attention_std = attention_stack.std(dim=0).mean(dim=[1, 2])
                    confidences = 1.0 / (1.0 + attention_std).cpu().numpy()
                else:
                    confidences = np.ones(len(predictions))
                
                predictions = predictions.cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_predictions.append(predictions)
            all_confidences.append(confidences)
        
        # Combine results
        return np.vstack(all_embeddings), np.concatenate(all_predictions), np.concatenate(all_confidences)
    
    try:
        # Process training data
        if len(train_df) > 10000:
            print("Processing training data in chunks...")
            train_embeddings, train_preds, train_confidences = process_df_in_chunks(train_df)
        else:
            # Convert features to float32
            X_train = np.array(train_df[features].fillna(0).values, dtype=np.float32)
            
            # Reshape data to [batch_size, num_features, 1]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                # Get model outputs
                train_preds, attentions = model(X_train_tensor)
                
                # Extract embeddings
                x = model.feature_embedding(X_train_tensor)
                x = x + model.pos_encoding[:, :X_train_tensor.size(1), :]
                
                for transformer in model.transformer_blocks:
                    x, _ = transformer(x)
                
                x = model.output_norm(x)
                train_embeddings = x.mean(dim=1).cpu().numpy()
                
                # Calculate confidence
                if attentions and len(attentions) > 0:
                    attention_stack = torch.stack(attentions)
                    attention_std = attention_stack.std(dim=0).mean(dim=[1, 2])
                    train_confidences = 1.0 / (1.0 + attention_std).cpu().numpy()
                else:
                    train_confidences = np.ones(len(train_preds))
                
                train_preds = train_preds.cpu().numpy()
        
        # Process validation data
        if len(val_df) > 10000:
            print("Processing validation data in chunks...")
            val_embeddings, val_preds, val_confidences = process_df_in_chunks(val_df)
        else:
            # Convert features to float32
            X_val = np.array(val_df[features].fillna(0).values, dtype=np.float32)
            
            # Reshape data to [batch_size, num_features, 1]
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                val_preds, attentions = model(X_val_tensor)
                
                # Extract embeddings
                x = model.feature_embedding(X_val_tensor)
                x = x + model.pos_encoding[:, :X_val_tensor.size(1), :]
                
                for transformer in model.transformer_blocks:
                    x, _ = transformer(x)
                
                x = model.output_norm(x)
                val_embeddings = x.mean(dim=1).cpu().numpy()
                
                # Calculate confidence
                if attentions and len(attentions) > 0:
                    attention_stack = torch.stack(attentions)
                    attention_std = attention_stack.std(dim=0).mean(dim=[1, 2])
                    val_confidences = 1.0 / (1.0 + attention_std).cpu().numpy()
                else:
                    val_confidences = np.ones(len(val_preds))
                
                val_preds = val_preds.cpu().numpy()
        
        # Create feature dataframes
        print("Creating feature dataframes...")
        # Create embedding feature columns
        train_features_dict = {}
        val_features_dict = {}
        
        # Add embedding features
        embed_dim = train_embeddings.shape[1]
        for i in range(embed_dim):
            col_name = f'af_emb_{i}'
            train_features_dict[col_name] = train_embeddings[:, i].astype(np.float32)
            val_features_dict[col_name] = val_embeddings[:, i].astype(np.float32)
        
        # Add prediction and confidence
        train_features_dict['prediction'] = train_preds.flatten().astype(np.float32)
        train_features_dict['af_confidence'] = train_confidences.flatten().astype(np.float32)
        train_features_dict['af_high_confidence'] = (train_confidences.flatten() > confidence_threshold).astype(np.float32)
        
        val_features_dict['prediction'] = val_preds.flatten().astype(np.float32)
        val_features_dict['af_confidence'] = val_confidences.flatten().astype(np.float32)
        val_features_dict['af_high_confidence'] = (val_confidences.flatten() > confidence_threshold).astype(np.float32)
        
        # Create dataframes
        train_features_df = pd.DataFrame(train_features_dict)
        val_features_df = pd.DataFrame(val_features_dict)
        
        print(f"Generated features - Train: {train_features_df.shape}, Val: {val_features_df.shape}")
        return train_features_df, val_features_df
    
    except Exception as e:
        print(f"Error generating features: {e}")
        print(traceback.format_exc())
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
