#!/bin/sh

# Exit on error
set -e

# Print header
printf "================================================================================\n"
printf "                          PyPotteryLens Setup\n"
printf "================================================================================\n\n"

# Function to check GPU availability
check_gpu() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi > /dev/null 2>&1; then
        printf "[✓] NVIDIA GPU detected\n"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
        printf "    • GPU: %s\n" "$GPU_NAME"
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        printf "    • CUDA Version: %s\n" "$CUDA_VERSION"
        return 0
    # Check for Apple Silicon (MPS)
    elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        printf "[✓] Apple Silicon detected - MPS (Metal Performance Shaders) support available\n"
        CHIP_INFO=$(system_profiler SPHardwareDataType | grep -E "Chip:|Apple" | head -1 | sed 's/^[[:space:]]*//')
        printf "    • %s\n" "$CHIP_INFO"
        return 0
    else
        printf "[!] No GPU detected - will install CPU-only version\n"
        return 1
    fi
}

# Check for required commands
if ! command -v python3 > /dev/null 2>&1; then
    printf "[!] Python 3 is required but not installed. Aborting.\n"
    exit 1
fi

if ! command -v pip3 > /dev/null 2>&1; then
    printf "[!] pip3 is required but not installed. Aborting.\n"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
printf "Detected Python version: %s\n" "$PYTHON_VERSION"
if [ "$PYTHON_VERSION" = "3.13" ]; then
    printf "[!] Warning: Python 3.13 detected. Some packages may have compatibility issues.\n"
    printf "    Scikit-image 0.24.0 is not compatible with Python 3.13.\n"
    printf "    The requirements will be updated to use compatible versions.\n"
fi

printf "Checking GPU...\n"
CUDA_AVAILABLE=0
GPU_TYPE="cpu"
if check_gpu; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        CUDA_AVAILABLE=1
        GPU_TYPE="cuda"
    elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        GPU_TYPE="mps"
    fi
fi

printf "\nChecking Python environment...\n"

# Check if Python virtual environment exists
VENV_EXISTS=0
if [ -d "venv" ]; then
    printf "[✓] Virtual environment already exists\n"
    VENV_EXISTS=1
else
    printf "[*] Creating virtual environment...\n"
    if ! python3 -m venv venv; then
        printf "[!] Failed to create virtual environment. Aborting.\n"
        exit 1
    fi
fi

# Activate virtual environment using . instead of source
printf "[*] Activating virtual environment...\n"
if ! . ./venv/bin/activate; then
    printf "[!] Failed to activate virtual environment. Aborting.\n"
    exit 1
fi

# Only install packages if venv is newly created
if [ $VENV_EXISTS -eq 0 ]; then
    printf "[*] New virtual environment detected, installing packages...\n"
    
    # Update pip first
    if ! python -m pip install --upgrade pip; then
        printf "[!] Failed to upgrade pip. Aborting.\n"
        exit 1
    fi
    
    # Install PyTorch based on CUDA availability
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        printf "[*] Installing PyTorch with CUDA support...\n"
        if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; then
            printf "[!] Failed to install PyTorch. Aborting.\n"
            exit 1
        fi
    else
        printf "[*] Installing CPU-only PyTorch...\n"
        if ! pip install torch torchvision torchaudio; then
            printf "[!] Failed to install PyTorch. Aborting.\n"
            exit 1
        fi
    fi
    
    # Install other requirements
    printf "[*] Installing base packages...\n"
    if ! pip install -r requirements.txt; then
        printf "[!] Failed to install requirements. Aborting.\n"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p models_vision models_classifier || {
    printf "[!] Failed to create model directories. Aborting.\n"
    exit 1
}

# Download models
printf "[*] Checking vision model...\n"
if ! python -c "from utils import download_model; exit(0 if download_model() else 1)"; then
    printf "[!] Error downloading vision model. Please check your internet connection and try again.\n"
    exit 1
fi

printf "[*] Checking classifier model...\n"
if ! python -c "from utils import download_model; exit(0 if download_model(url='https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/model_classifier.pth', dest_path='models_classifier/model_classifier.pth') else 1)"; then
    printf "[!] Error downloading classifier model. Please check your internet connection and try again.\n"
    exit 1
fi

# Verify installation
printf "\nVerifying PyTorch installation...\n"
if ! python -c "
import torch
print(f'[✓] PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'[✓] CUDA available: True')
    print(f'[✓] CUDA Device: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print(f'[✓] MPS (Metal) available: True')
    print(f'[✓] Device: Apple Silicon GPU')
else:
    print(f'[✓] Running on: CPU')
"; then
    printf "[!] Failed to verify PyTorch installation. Aborting.\n"
    exit 1
fi

# Start the application
printf "\n[*] Starting PyPotteryLens...\n"
if ! python app.py; then
    printf "\n[!] An error occurred while starting PyPotteryLens. Please check the messages above.\n"
    printf "Press Enter to continue..."
    read answer
    exit 1
fi

printf "Press Enter to continue..."
read answer