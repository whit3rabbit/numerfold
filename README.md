
# NumerAIFold: AlphaFold-Inspired Pipeline for Numerai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Development Status](https://img.shields.io/badge/development%20status-alpha-orange.svg)]()

An AlphaFold-inspired approach to Numerai predictions, implementing concepts from protein structure prediction for financial forecasting.

*This project is currently just for fun. It is not to be taken seriously but feel free to adapt it or adjust it.*

## Concept: From Protein Folding to Financial Prediction

NumerAIFold adapts AlphaFold's protein structure prediction concepts to financial markets:

| AlphaFold Concept | NumerAIFold Adaptation |
|-------------------|------------------------|
| Amino Acid Sequence | Feature Vector Sequence |
| Protein Domains | Feature Domains (Related Feature Groups) |
| Evolutionary Profile | Era-wise Feature Stability |
| Residue Contacts | Feature Interactions/Correlations |
| Structure Confidence | Prediction Confidence Score |
| MSA Transformer | Feature Domain Transformer |
| Template Processing | Historical Pattern Analysis |

## Workflow

```
                                     ┌─────────────────────┐
                                     │                     │
                                     │  Numerai Features   │
                                     │                     │
                                     └─────────┬───────────┘
                                               │
                                               ▼
┌─────────────────────┐            ┌─────────────────────┐            ┌─────────────────────┐
│    Domain Phase     │            │   Training Phase     │            │  Prediction Phase   │
│                     │            │                     │            │                     │
│  ┌───────────────┐  │            │  ┌───────────────┐  │            │  ┌───────────────┐  │
│  │Feature Domain │  │            │  │  Transform &   │  │            │  │  Generate     │  │
│  │Identification │  │            │  │   Attention    │  │            │  │  Confidence   │  │
│  └───────┬───────┘  │            │  └───────┬───────┘  │            │  └───────┬───────┘  │
│          │          │            │          │          │            │          │          │
│  ┌───────┴───────┐  │            │  ┌───────┴───────┐  │            │  ┌───────┴───────┐  │
│  │  Stability    │  │──────────▶│  │  Train Model  │  │──────────▶│  │   Weighted    │  │
│  │   Analysis    │  │            │  │              │  │            │  │  Predictions  │  │
│  └───────┬───────┘  │            │  └───────┬───────┘  │            │  └───────┬───────┘  │
│          │          │            │          │          │            │          │          │
│  ┌───────┴───────┐  │            │  ┌───────┴───────┐  │            │  ┌───────┴───────┐  │
│  │  Evolution    │  │            │  │  Confidence   │  │            │  │   Evaluate    │  │
│  │   Profiles    │  │            │  │  Calibration  │  │            │  │    Results    │  │
│  └───────────────┘  │            │  └───────────────┘  │            │  └───────────────┘  │
│                     │            │                     │            │                     │
└─────────────────────┘            └─────────────────────┘            └─────────────────────┘
                                               │
                                               ▼
                                     ┌─────────────────────┐
                                     │   Final Pipeline    │
                                     │     Evaluation      │
                                     └─────────────────────┘

```


### Core Components

#### NumerAIFold Model (`core/model.py`)

- Transformer-based architecture with self-attention mechanisms
- Learnable positional encodings for feature order preservation
- Multiple transformer blocks for deep feature interaction learning
- Confidence score generation based on attention patterns

#### Feature Processing (`features/`)

- Feature stability analysis across eras
- AlphaFold-inspired feature engineering pipeline
- Domain-based feature processing
- TODO: Feature reconstruction objectives

#### Domain Analysis (`domains/`)

- Feature domain identification using clustering
- Domain relationship analysis
- Interactive visualizations for domain exploration
- Domain-specific evolutionary profiles

### Pipeline Phases

#### Phase 1: Feature Domain Identification

```
Raw Features → Domain Clustering → Stability Analysis → Domain Relationships
```

- Implemented in `domains/identification.py`
- Uses PCA and UMAP for dimensionality reduction
- KMeans clustering for domain assignment
- Generates domain visualizations and relationship analyses

#### Phase 2: Model Architecture & Training

```
NumerAIFold Model → Training Loop → Confidence Calibration
```

- Transformer-based architecture in `core/model.py`
- Training implementation in `core/training.py`
- Includes early stopping and learning rate scheduling
- TODO: Multi-task training objectives
- TODO: Market regime detection

#### Phase 3: Prediction Generation

```
Feature Processing → Confidence-Weighted Predictions → Evaluation
```

- Prediction generation in `core/evaluation.py`
- Confidence score computation
- Comprehensive evaluation metrics
- TODO: Ensemble integration
- TODO: Regime-specific adjustments

## Key Features

### Implemented

- **Domain Identification**: Automatic feature domain discovery
- **Stability Analysis**: Era-wise feature stability metrics
- **Attention Mechanisms**: Both standard and pairwise attention implementations
- **Confidence Scoring**: Based on attention pattern consistency
- **Evolutionary Profiles**: Feature behavior analysis across eras
- **Visualization Tools**: Domain visualization and analysis tools

## Usage

### Basic Usage

```python
from numeraifold.pipeline import run_alphafold_pipeline
from numeraifold.data import load_data

# Load data
train_df, val_df, features, targets = load_data()

# Run pipeline
results = run_alphafold_pipeline(
    train_df=train_df,
    val_df=val_df,
    features=features,
    targets=targets
)
```

### Domains-Only Analysis

```python
from numeraifold.pipeline import run_domains_only_pipeline

results = run_domains_only_pipeline(
    data_version="v5.0",
    feature_set="small"
)
```

## Configuration

Key configuration options in `config.py`:

```python
DATA_VERSION = "v5.0"
FEATURE_SET = "small"
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics including:

- Mean correlation across eras
- Correlation stability (standard deviation)
- Sharpe ratio of era-wise correlations
- Feature neutral correlation
- Confidence-weighted metrics

## Development

- The codebase follows a modular structure
- Each component is designed to be independently testable
- Configuration management through `pipeline/configuration.py`
- Extensive error handling and validation throughout

## Future Enhancements

See TODO.md

## Installation

```bash
pip install -e "git+https://github.com/whit3rabbit/numerfold.git"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
