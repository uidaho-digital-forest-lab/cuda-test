# Basic CUDA test

## Background

This project provides a simple CUDA test script and supporting infrastructure for verifying GPU availability and basic compute functionality across a variety of execution environments.

### Testing CUDA

`cuda_test.py` is a minimal PyTorch-based script that checks whether CUDA is available, reports on any detected GPUs (name, VRAM, compute capability), and runs a small matrix multiply on each GPU as a functional smoke test. It serves as a "hello world" for GPU compute; if this script passes, the CUDA stack is working end-to-end.

### Testing CUDA with Juice

[Juice](https://juicelabs.co) is GPU-over-IP software that allows applications to use remote GPUs over a standard TCP/IP network. It works by injecting a stub `libcuda.so` via `LD_PRELOAD` that transparently proxies CUDA calls to a remote GPU agent, with no modifications required to application code. Running `cuda_test.py` via `juice run` verifies that a Juice pool is reachable, a GPU session can be allocated, and CUDA compute works over the network.

### Testing CUDA with Apptainer

[Apptainer](https://apptainer.org) is a container runtime designed for HPC
environments. `cuda_test.def` defines an Apptainer container image (SIF) with a self-contained Python/PyTorch environment. Running `cuda_test.py` inside the container via `apptainer run --nv` verifies that the container build is correct and complete and that Apptainer's NVIDIA GPU passthrough is working on the host.

### Testing CUDA with Apptainer + Juice

Combining Apptainer and Juice requires some care. Juice injects its stub `libcuda.so` via `LD_PRELOAD` into every process it spawns, including the Apptainer container runtime. The stub has dependencies on other Juice libraries that are not present inside the container. The solution is to bind-mount the Juice installation directory into the container and expose it via `APPTAINERENV_LD_LIBRARY_PATH`, allowing the dynamic linker inside the container to resolve Juice's library dependencies. Successfully running this demonstrates the ability to containerize scripts and execute them using the Juice GPU.

## Setup/Requirements/Notes
- Clone this repo
- Download/install the [Juice software](https://app.juicelabs.co/) for your OS
- Install apptainer in linux
- For linux usage, put the juice software in a `juice` subdirectory of your repo clone
- Although we build and activate a conda environment for the first vignettes, it is not needed under apptainer execution as `cuda_test.sif` contains the necessary environment
- `cuda_test.sh` is a helper script that runs the whole final Apptainer + Juice setup, including Juice login and pool selection:
```Usage:
  ./juice_cuda_test.sh                        # interactive pool selection
  ./juice_cuda_test.sh <pool-id>              # use a specific pool ID
  JUICE_POOL_ID=<pool-id> ./juice_cuda_test.sh
```

## Vignettes
### Basic CUDA test (windows)

1. Activate the cuda-test environment and run the script:
```
conda env create -f cuda_env.yml
conda activate cuda-test
python cuda_test.py
```

2. You should see a result from `cuda_test.py` from the local GPUs:
```
Python version:  3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:36:00) [MSC v.1944 64 bit (AMD64)]
PyTorch version: 2.6.0+cu124

CUDA available:  True
GPU count:       2

GPU 0: Quadro RTX 5000
  VRAM:            16.0 GB
  CUDA capability: 7.5
  Multiprocessors: 48

GPU 1: Quadro RTX 5000
  VRAM:            16.0 GB
  CUDA capability: 7.5
  Multiprocessors: 48

Current device:  0
cuDNN enabled:   True
cuDNN version:   90100

--- Tensor smoke test ---
GPU 0: matrix multiply (1000x1000): OK  ->  shape (1000, 1000), device=cuda:0
GPU 1: matrix multiply (1000x1000): OK  ->  shape (1000, 1000), device=cuda:1

All checks passed - CUDA OK
```

### Basic CUDA test with Juice (windows)

1. Activate the cuda-test environment:
```
conda env create -f cuda_env.yml
conda activate cuda-test
```

2. Prepare juice - follow the prompts to authenticate:
```
& 'C:\Program Files\Juice GPU\juice.exe' login
```

3. View the juice pool list. The Personal Pool contains one juice agent, linuxviz (172.28.53.25):
```
& 'C:\Program Files\Juice GPU\juice.exe' pool list
```

Juice GPU, v30.0.0-50
| Org-Name | Name | ID | Sessions [Active/Total] | GPUs | Share | Use |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| University of Idaho - College of Natural Resources | Personal Pool [flathers@uidaho.edu] | 956dacfa-5137-4126-9856-198a6b593fea | 0 / 0 | 1 | ✔ | ✔|


4. Use your pool ID to run `cuda_test.py`:
```
'C:\Program Files\Juice GPU\juice.exe' run --pool-ids 956dacfa-5137-4126-9856-198a6b593fea python cuda_test.py

Juice GPU, v30.0.0-50

2026/04/16 15:47:22 Waiting for C:\Users\flathers\AppData\Local\Juice GPU\logs\client-07f6b20f-f34b-441a-bc7b-7c4924da22d5.log to appear...
Python version:  3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:36:00) [MSC v.1944 64 bit (AMD64)]
PyTorch version: 2.6.0+cu124

CUDA available:  True
GPU count:       1

GPU 0: Juice GPU [NVIDIA GeForce RTX 4090]
  VRAM:            24.0 GB
  CUDA capability: 8.9
  Multiprocessors: 128

Current device:  0
cuDNN enabled:   True
cuDNN version:   90100

--- Tensor smoke test ---
GPU 0: matrix multiply (1000x1000): OK  ->  shape (1000, 1000), device=cuda:0

All checks passed - CUDA OK
```

We see one GeForce RTX 4090 on this run, which is correct for this pool. We successfully ran CUDA on a remote host using Juice.

### Basic CUDA test using apptainer (linux)
1. Build the apptainer image. This will take a minute. The linux host must have apptainer installed:
```
apptainer build --fakeroot cuda_test.sif cuda_test.def
```

2. Invoke the script via apptainer (on a linux host equipped with a CUDA GPU):
```
apptainer run --nv cuda_test.sif
```

```
Python version:  3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]
PyTorch version: 2.6.0+cu124

CUDA available:  True
GPU count:       1

GPU 0: NVIDIA GeForce RTX 4090
  VRAM:            23.5 GB
  CUDA capability: 8.9
  Multiprocessors: 128

Current device:  0
cuDNN enabled:   True
cuDNN version:   90100

--- Tensor smoke test ---
GPU 0: matrix multiply (1000x1000): OK  ->  shape (1000, 1000), device=cuda:0

All checks passed - CUDA OK
```

### Basic CUDA test using apptainer and Juice (linux)
1. Prepare juice - follow the prompts to authenticate:
```
./juice/juice login
```

2. This assumes the apptainer image has been built. If not, see above. Invoke the script via apptainer with Juice wrapper. We must also bind mount the juice library path and feed it into the apptainer environment:
```
APPTAINERENV_LD_LIBRARY_PATH=/juice:/juice/compute:/juice/graphics ./juice/juice run --pool-ids 956dacfa-5137-4126-9856-198a6b593fea apptainer run --bind ./juice:/juice cuda_test.sif
```
```
Juice GPU, v30.0.0-50

2026/04/17 15:25:51 Waiting for /mnt/ceph/flathers/.config/Juice GPU/logs/client-eda7e513-c878-4996-81d3-4ed3e72f07d1.log to appear...
Python version:  3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0]
PyTorch version: 2.6.0+cu124

CUDA available:  True
GPU count:       1

GPU 0: Juice GPU [NVIDIA GeForce RTX 4090]
  VRAM:            24.0 GB
  CUDA capability: 8.9
  Multiprocessors: 128

Current device:  0
cuDNN enabled:   True
cuDNN version:   90100

--- Tensor smoke test ---
GPU 0: matrix multiply (1000x1000): OK  ->  shape (1000, 1000), device=cuda:0

All checks passed - CUDA OK
```

We see one GeForce RTX 4090 on this run, which is correct for this pool. We successfully ran CUDA on a remote host using apptainer and Juice.


