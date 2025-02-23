
# NumerAIFold: AlphaFold-Inspired Pipeline for Numerai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Development Status](https://img.shields.io/badge/development%20status-alpha-orange.svg)]()

## Description

NumerAIFold is a Python project that adapts concepts from AlphaFold, a groundbreaking protein structure prediction model, to the Numerai Tournament. It aims to improve prediction accuracy by:

1.  **Identifying Feature Domains:**  The pipeline clusters Numerai's features into "domains" of related features using dimensionality reduction (PCA and UMAP) and clustering (KMeans).
2.  **Building a Transformer Model:** A custom transformer model, inspired by AlphaFold's architecture, is trained to capture complex relationships within and between these feature domains.
3.  **Generating AlphaFold-Inspired Features:**  The model generates embeddings and confidence scores, which are used as additional features.
4.  **Performing Evolutionary Analysis:**  The project includes functions to analyze feature stability and evolution across eras, similar to how AlphaFold analyzes evolutionary conservation in protein sequences.

This project is currently in **alpha** status.  It's a research project and is not guaranteed to produce winning submissions in the Numerai Tournament.

**1. `pip install` Instructions**

We'll add instructions for installing directly from GitHub using `pip`. This is the preferred method for most users, as it handles dependencies automatically. We'll also cover the case where someone clones the repository and wants to install locally.

**2. Relative Imports: Explanation and (Possible) Change**

*   **The Issue:** Your current project structure uses relative imports (e.g., `from .core import model` within files in the `src/numeraifold` directory). This works fine when you're *inside* the `src` directory (as we did with `%cd numeraifold/src` in the Colab example). However, it creates problems if someone tries to import your package *after* installing it with `pip`.  The installed package won't have the same relative structure.

*   **Two Solutions (You need to choose ONE):**

    *   **Solution 1 (Recommended - Keep Relative Imports, Use Editable Install):**  Keep the relative imports *and* instruct users to install in "editable" mode (`pip install -e .`).  This is the best option for development, and it's also suitable for users who want to modify the code.  It creates a symbolic link to your project directory, so changes you make are immediately reflected without reinstalling.

    *   **Solution 2 (Change to Absolute Imports):**  Change all your relative imports to absolute imports (e.g., `from numeraifold.core import model`). This is cleaner for standard installation (`pip install .`), but it makes development slightly more cumbersome, because you need to reinstall the package after every code change.

    **I strongly recommend Solution 1 (editable install) for this project.** It's the standard practice for packages under development, and it's more flexible for users.  I'll update the `README.md` with this approach.

**3. ASCII Directory Tree**

We'll add a clear ASCII representation of your project's directory structure to the `README.md`. This helps users understand how the code is organized.

**Updated `README.md`**

```markdown
# NumerAIFold: AlphaFold-Inspired Pipeline for Numerai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Development Status](https://img.shields.io/badge/development%20status-alpha-orange.svg)]()

## Description

NumerAIFold is a Python project that adapts concepts from AlphaFold, a groundbreaking protein structure prediction model, to the Numerai Tournament. It aims to improve prediction accuracy by:

1.  **Identifying Feature Domains:**  The pipeline clusters Numerai's features into "domains" of related features using dimensionality reduction (PCA and UMAP) and clustering (KMeans).
2.  **Building a Transformer Model:** A custom transformer model, inspired by AlphaFold's architecture, is trained to capture complex relationships within and between these feature domains.
3.  **Generating AlphaFold-Inspired Features:**  The model generates embeddings and confidence scores, which are used as additional features.
4.  **Performing Evolutionary Analysis:**  The project includes functions to analyze feature stability and evolution across eras, similar to how AlphaFold analyzes evolutionary conservation in protein sequences.

This project is currently in **alpha** status.  It's a research project and is not guaranteed to produce winning submissions in the Numerai Tournament.

## Installation

### Option 1: Install Directly from GitHub (Recommended)

This method uses `pip` to install the package directly from the GitHub repository.  It's the easiest way to get started.  We use an "editable" install (`-e`) so that any changes you make to the code are immediately reflected.

```bash
pip install -e "git+https://github.com/whit3rabbit/numeraifold.git#egg=numeraifold"
```

### Option 2: Clone and Install Locally

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/whit3rabbit/numeraifold.git
    cd numeraifold
    ```

2.  **Install in Editable Mode:**

    ```bash
    pip install -e .
    ```
    The `-e .` installs the package in "editable" mode. Changes to files in the `src/` directory will be reflected immediately without needing to reinstall.

### Option 3: Google Colab (Easiest for Quick Start)

The easiest way to run NumerAIFold without any local setup is within a Google Colab notebook:

1.  **Open a New Colab Notebook:** Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

2. **Run this code:**

```python
# --- Cell 1: Clone, Install, and Change Directory ---
!git clone https://github.com/whit3rabbit/numeraifold.git
!pip install -e numeraifold/
%cd numeraifold/src

# --- Cell 2: Import and Run ---
import torch
import numpy as np
from numeraifold.pipeline import run_alphafold_pipeline, run_domains_only_pipeline
from numeraifold.data.loading import load_data
from numeraifold.utils.seed import set_seed

# Set seed and check CUDA
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
print(f"CUDA available: {torch.cuda.is_available()}")

# --- Cell 3: Run Domains-Only Pipeline (Optional) ---
results = run_domains_only_pipeline()
if results and 'error' not in results:
    print(f"Domains saved to: {results.get('domains_saved_path', 'N/A')}")
else:
    print(f"Domain extraction failed: {results.get('error', 'Unknown error')}")

# --- Cell 4: Load Data and Run Full Pipeline ---
train_df, val_df, features, targets = load_data(data_version="v5.0", feature_set="small")

if train_df is not None:
    results = run_alphafold_pipeline(
        train_df=train_df,
        val_df=val_df,
        features=features,
        targets=targets,
        epochs=3,          # Reduced for demonstration
        batch_size=64,
        n_clusters=10,
        save_model = False, # Change to True to save
        skip_phase1 = True  # Load domain data from cache
    )

    if 'results_standard' in results:
        print("\nStandard Prediction Metrics:")
        print(results['results_standard'])

    if 'results_weighted' in results:
        print("\nConfidence-Weighted Prediction Metrics:")
        print(results['results_weighted'])
    if 'evaluation' in results:
      if 'val_df' in results['evaluation']:
            print("\nValidation Dataframe (First 5 rows):")
            print(results['evaluation']['val_df'].head())
else:
    print("Data loading failed.")

# --- Cell 5: Further Analysis (Optional) ---
if 'feature_groups' in results:
    print("\nFeature Groups:")
    for domain, features_in_domain in results['feature_groups'].items():
        print(f"- {domain}: {len(features_in_domain)} features")

if 'stability_df' in results:
    print("\nFeature Stability (First 5 rows):")
    print(results['stability_df'].head())

# Example: Access the trained model
if 'trained_model' in results and results['trained_model'] is not None:
    model = results['trained_model']

```

## Directory Structure

```
numeraifold/
├── README.md         <- This file
├── requirements.txt  <- Python dependencies
├── setup.py          <- Installation script
├── pyproject.toml   <- Build system configuration
└── src/
    └── numeraifold/
        ├── __init__.py
        ├── config.py       <- Configuration settings
        ├── core/
        │   ├── __init__.py
        │   ├── evaluation.py <- Model evaluation functions
        │   ├── model.py      <- Transformer model definition
        │   ├── training.py   <- Model training loop
        ├── data/
        │   ├── __init__.py
        │   ├── dataloader.py <- PyTorch DataLoader utilities
        │   ├── loading.py    <- Data loading and preprocessing
        │   └── preprocessing.py <- Data preprocessing functions
        ├── domains/
        │   ├── __init__.py
        │   ├── analysis.py     <- Feature domain analysis
        │   ├── identification.py <- Feature domain identification
        │   └── visualization.py  <- Domain visualization tools
        ├── features/
        │   ├── __init__.py
        │   ├── engineering.py  <- Feature engineering
        │   ├── sequences.py    <- Sequence representation creation
        │   └── stability.py    <- Feature stability analysis
        ├── pipeline/
        │   ├── __init__.py
        │   ├── configuration.py <- Pipeline configuration
        │   └── execution.py    <- Main pipeline execution
        └── utils/
            ├── __init__.py
            ├── artifacts.py   <- Model and data saving/loading
            ├── domain.py      <- Domain data integration utilities
            ├── seed.py        <- Random seed setting
            └── visualization.py <- General visualization utilities
```

## Usage

After installation, you can import and use the modules in your Python scripts or notebooks.  The Colab example above provides a good starting point.  The main functions are:

*   `numeraifold.pipeline.run_alphafold_pipeline()`: Runs the complete pipeline.
*   `numeraifold.pipeline.run_domains_only_pipeline()`: Runs only the feature domain identification.
*   `numeraifold.data.loading.load_data()`: Loads the Numerai data.

## Contributing

Contributions are welcome!  Please see the [issues](https://github.com/whit3rabbit/numeraifold/issues) page for open tasks and bug reports.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
