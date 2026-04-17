import torch
import sys

print(f"Python version:  {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print()

# --- CUDA availability ---
cuda_available = torch.cuda.is_available()
print(f"CUDA available:  {cuda_available}")

if not cuda_available:
    print("\nNo CUDA-capable GPU detected.")
    print("Possible reasons:")
    print("  - No NVIDIA GPU present")
    print("  - CUDA drivers not installed / outdated")
    print("  - PyTorch was installed without CUDA support")
    sys.exit(0)

# --- GPU details ---
device_count = torch.cuda.device_count()
print(f"GPU count:       {device_count}")

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  VRAM:            {props.total_memory / 1024**3:.1f} GB")
    print(f"  CUDA capability: {props.major}.{props.minor}")
    print(f"  Multiprocessors: {props.multi_processor_count}")

print(f"\nCurrent device:  {torch.cuda.current_device()}")
print(f"cuDNN enabled:   {torch.backends.cudnn.enabled}")
print(f"cuDNN version:   {torch.backends.cudnn.version()}")

# --- Functional test: tensor on each GPU ---
print("\n--- Tensor smoke test ---")
for i in range(device_count):
    device = f"cuda:{i}"
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    print(f"GPU {i}: matrix multiply (1000x1000): OK  ->  shape {tuple(c.shape)}, device={c.device}")

print("\nAll checks passed - CUDA OK")
