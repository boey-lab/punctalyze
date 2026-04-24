### commands for bioimage-fast build, run one line at a time in your terminal
```bash
conda create -y -n bioimage-fast -c conda-forge python=3.12 
conda activate bioimage-fast
```
### for windows users 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### for mac users 
```bash
conda install pytorch torchvision torchaudio -c pytorch-nightlyconda install conda-forge::ipykernel jupyter matplotlib
conda install conda-forge::numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install conda-forge::napari matplotlib-scalebar
conda install conda-forge::seaborn statannotations
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi bioio-nd2
conda install numpy=1.26 # later had to run this due to numpy issues
```

### homebrew help for macos
1. Install Homebrew from your Mac terminal software (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Use Homebrew to install OpenSSL:
```bash
brew install openssl
```
3. Open a new Terminal window, activate your conda environment (if using one):
```bash
conda activate <your_env_name>
```
4. Set environment variables so pip can find OpenSSL (run these in the same terminal before pip install)
```bash
export OPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl
export LDFLAGS="-L/opt/homebrew/opt/openssl/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openssl/include"
```
5. Now install BioIO and all plugins:
```bash
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi
```
6. If you see dependency conflict warnings about aicsimageio, you can ignore them as long as you are using BioIO for your image reading/writing


### want to check your macos cellpose gpu is working
```bash
"""
Cellpose GPU detection — copy-paste and run.
Checks CUDA (NVIDIA) → MPS (Apple Silicon) → CPU fallback.
"""
import torch
from cellpose import core, models


def detect_cellpose_device(verbose=True):
    """
    Detect the best available device for Cellpose.

    Returns:
        device (torch.device): the device Cellpose will use
        gpu_available (bool): True if GPU (CUDA or MPS) is usable
        backend (str): 'cuda' | 'mps' | 'cpu'
    """
    # Cellpose's own check — actually allocates a test tensor to verify
    gpu_available = core.use_gpu()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        backend = "cuda"
        info = f"CUDA GPU: {torch.cuda.get_device_name(0)} " \
               f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "mps"
        info = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        backend = "cpu"
        info = "CPU only (no GPU detected)"

    if verbose:
        print("=" * 60)
        print("Cellpose Device Detection")
        print("=" * 60)
        print(f"PyTorch version : {torch.__version__}")
        print(f"GPU available   : {gpu_available}")
        print(f"Selected device : {device}")
        print(f"Backend         : {backend}")
        print(f"Hardware        : {info}")
        print("=" * 60)

    return device, gpu_available, backend


def verify_cellpose_on_gpu():
    """Actually load Cellpose and confirm the model sits on GPU."""
    device, gpu_available, backend = detect_cellpose_device()

    print("\nLoading Cellpose model...")
    model = models.CellposeModel(gpu=gpu_available)

    actual_device = next(model.net.parameters()).device
    print(f"Cellpose model is on: {actual_device}")

    if gpu_available and actual_device.type in ("cuda", "mps"):
        print("✓ GPU acceleration is ACTIVE")
    else:
        print("✗ Running on CPU (GPU not used)")

    return model


if __name__ == "__main__":
    model = verify_cellpose_on_gpu()
```
