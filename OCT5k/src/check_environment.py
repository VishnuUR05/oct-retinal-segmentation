import sys
import torch
import torchvision

def main():
    print("="*40)
    print("Environment Verification")
    print("="*40)
    print(f"Python version:       {sys.version.split(' ')[0]}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Torchvision version:  {torchvision.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA availability:    {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version:         {torch.version.cuda}")
        print(f"GPU name:             {torch.cuda.get_device_name(0)}")
        
        # GPU Memory
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        print(f"GPU memory:           {total_memory_gb:.2f} GB")
        
        print(f"cuDNN availability:   {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN version:        {torch.backends.cudnn.version()}")
    else:
        print("CUDA is NOT available. PyTorch cannot use the GPU.")
        
    print("="*40)

if __name__ == "__main__":
    main()
