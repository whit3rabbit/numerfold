import psutil
import os

def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    print(f"{prefix} Memory usage: {mem:.2f} GB")