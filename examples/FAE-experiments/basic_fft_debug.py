import torch

def basic_fft_debug(device):
    # Create a simple 1D tensor
    x = torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=torch.float32)


    # Move tensor to the specified device
    x = x.to(device)

    # Verify the device of the tensor
    print(f"Tensor device: {x.device}")
    
    # Perform FFT
    fft_result = torch.fft.fft(x)

    # Print the results
    print("Input Tensor:", x)
    print("FFT Result:", fft_result)

    # Perform Inverse FFT
    ifft_result = torch.fft.ifft(fft_result)

    # Print the inverse FFT results
    print("Inverse FFT Result:", ifft_result)


if __name__ == "__main__":
     # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Get number of GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    device = None
    # Get GPU name(s)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Check memory
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        device = torch.device("cuda")
    else:
        print("WARNING: No GPU detected! Running on CPU")
        device = torch.device("cpu")


    basic_fft_debug(device)