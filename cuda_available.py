import torch

def is_cuda_available():
    """
    Check if CUDA is available and return the appropriate device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
if __name__ == "__main__":
    device = is_cuda_available()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA is available with {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, using CPU.")