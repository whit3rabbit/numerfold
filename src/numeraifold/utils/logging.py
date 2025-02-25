import psutil
import os
import torch
import gc

def log_memory_usage(label="Current memory usage"):
    """
    Log the current memory usage of the process.
    
    Args:
        label: Label to prefix the memory usage log
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    # Get GPU memory usage if available
    gpu_memory_usage = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_memory_usage = f"Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB"
        except:
            pass
    
    print(f"{label} - RAM: {memory_usage_mb:.2f} MB, GPU: {gpu_memory_usage}")


def optimize_memory():
    """
    Perform memory optimization steps to free up memory.
    This includes garbage collection and clearing PyTorch's CUDA cache.
    """
    # Collect garbage
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    log_memory_usage("After memory optimization")
