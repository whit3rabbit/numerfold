import torch

# Data configuration
DATA_VERSION = "v5.0"
FEATURE_SET = "small"

# Model configuration
MAIN_TARGET = "target"
AUX_TARGETS = []
NUM_AUX_TARGETS = 5
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default paths
BASE_PATH = "."
DOMAINS_SAVE_PATH = "feature_domains_data.csv"

# Model architecture defaults
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_DROPOUT = 0.1

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 3

# Feature domain defaults
DEFAULT_N_CLUSTERS = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.5