import torch

def print_gpu_memory():
     """Print current GPU memory usage in GB
     
        This function checks if a CUDA-enabled GPU is available and prints the allocated and reserved memory in gigabytes (GB).

        Returns:
            None
     """
     if torch.cuda.is_available():
          allocated = torch.cuda.memory_allocated() / 1024**3
          reserved = torch.cuda.memory_reserved() / 1024**3
          print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")