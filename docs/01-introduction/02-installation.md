# Installation

## System Requirements

The project requires Python 3.10 or higher. Earlier versions are not compatible due to type dependencies and language features used in the codebase.

For NVIDIA GPUs, CUDA 12.9 or higher is required. The code uses CUDA features that are not available in earlier versions. If you have an older GPU with CUDA 11.8, you will need to modify the build scripts or use the pure Python implementation, which is slower but functional.

The minimum recommended RAM is 16 GB for running medium-scale experiments. Experiments with large models or large batches may require 32 GB or more, especially during CUDA kernel compilation.

## Core Dependencies

The project depends on PyTorch 2.0 or higher for automatic differentiation and tensor operations. We install torch with CUDA support for GPU, but the code also runs on CPU with significant performance degradation.

We need einops for efficient tensor manipulation and einops.layers for layers with simplified notation. Most reshaping and transposition operations depend on this library.

For visualization and debugging, we install matplotlib and seaborn. These dependencies are optional if you only need to train models without visualizing results.

## Virtual Environment

Creating a virtual environment is highly recommended to isolate project dependencies.

```bash
python -m venv manifold-env
source manifold-env/bin/activate  # Linux/Mac
# o
manifold-env\Scripts\activate  # Windows
```

On Windows, activation may require administrator permissions depending on security settings. If it fails, run the terminal as administrator.

## Step-by-Step Installation

First, clone the repository and navigate to the project directory.

```bash
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold
```

Second, install the Python dependencies. The requirements.txt file includes all required dependencies.

```bash
pip install -r requirements.txt
```

Third, if you have an NVIDIA GPU and want acceleration, compile the CUDA kernels. This step is optional but recommended for performance.

```bash
python -m gfn.cuda.precompile_kernels
```

The compilation process can take several minutes depending on your GPU. If you see compilation errors, verify that CUDA is properly installed and that your GPU is compatible.

## Installation Verification

Run the consistency tests to verify that the installation is correct.

```bash
python tests/test_cuda_python_consistency.py
```

This test verifies that the Python implementation produces the same results as the CUDA implementation. If the test fails, the system can still function, but with degraded performance.

For a more basic verification, run the quick start script.

```bash
python demos/tinystories/train_tinystories.py --config configs/training/experiment_overfit_10k.yaml --steps 100
```

If the script runs without errors and reports loss, the basic installation is correct.

## Common Issues

If you import the gfn module and receive an error about torch, verify that PyTorch is installed correctly.

```bash
python -c "import torch; print(torch.__version__)"
```

If the version is earlier than 2.0, update PyTorch with pip install torch --upgrade.

If CUDA compilation fails with include errors, verify that the CUDA toolkit is in your PATH. On Linux, it typically needs to be in /usr/local/cuda.

If parity tests fail with small differences, this may be expected depending on your GPU. Differences smaller than 1e-5 are generally acceptable.

## GPU Configuration

For optimal performance, make sure CUDA dominates GPU memory allocation before training. The following script configures the environment.

```bash
export PYTORCH_CUDA_ALLOCATOR=max
export CUDA_LAUNCH_BLOCKING=0
```

These environment variables improve memory allocation and reduce synchronization overhead.

If you have multiple GPUs, you can specify which one to use with export CUDA_VISIBLE_DEVICES=0 to use only the first GPU.

---

**Manifold Labs (Joaquín Stürtz)**
