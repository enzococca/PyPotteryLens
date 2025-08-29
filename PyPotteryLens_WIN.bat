@echo off
echo.
echo ================================================================================
echo                          PyPotteryLens Setup
echo ================================================================================
echo.

:: Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%
if "%PYTHON_VERSION%"=="3.13" (
    echo [!] Warning: Python 3.13 detected. Some packages may have compatibility issues.
    echo     Scikit-image 0.24.0 is not compatible with Python 3.13.
    echo     The requirements will be updated to use compatible versions.
)

:: Check CUDA availability through nvidia-smi
echo Checking GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] NVIDIA GPU detected
    for /f "tokens=2 delims=," %%a in ('nvidia-smi --query-gpu^=name --format^=csv ^| findstr /v "name"') do set GPU_NAME=%%a
    echo     • GPU: %GPU_NAME%
    
    :: Get CUDA version
    for /f "tokens=3" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%a
    echo     • CUDA Version: %CUDA_VERSION%
    set CUDA_AVAILABLE=1
) else (
    echo [!] No NVIDIA GPU detected - will install CPU-only version
    set CUDA_AVAILABLE=0
)

echo.
echo Checking Python environment...

:: Check if Python virtual environment exists
set VENV_EXISTS=0
if exist "venv" (
    echo [*] Virtual environment already exists
    set VENV_EXISTS=1
) else (
    echo [*] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to create virtual environment. Aborting.
        pause
        exit /b 1
    )
    set VENV_EXISTS=0
)

:: Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [!] Failed to activate virtual environment. Aborting.
    pause
    exit /b 1
)

:: Only install packages if venv is newly created
if %VENV_EXISTS%==0 (
    echo [*] New virtual environment detected, installing packages...
    
    :: Update pip first
    echo [*] Upgrading pip...
    python -m pip install --upgrade pip
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to upgrade pip. Aborting.
        pause
        exit /b 1
    )
    
    :: Install PyTorch based on CUDA availability
    if %CUDA_AVAILABLE%==1 (
        echo [*] Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if %ERRORLEVEL% NEQ 0 (
            echo [!] Failed to install PyTorch. Aborting.
            pause
            exit /b 1
        )
    ) else (
        echo [*] Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
        if %ERRORLEVEL% NEQ 0 (
            echo [!] Failed to install PyTorch. Aborting.
            pause
            exit /b 1
        )
    )
    
    :: Install other requirements
    echo [*] Installing base packages...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to install requirements. Aborting.
        pause
        exit /b 1
    )
)

:: Create necessary directories
echo [*] Creating model directories...
if not exist "models_vision" mkdir models_vision
if not exist "models_classifier" mkdir models_classifier

:: Download models
echo [*] Checking vision model...
python -c "from utils import download_model; exit(0 if download_model() else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Error downloading vision model. Please check your internet connection and try again.
    pause
    exit /b 1
)

echo [*] Checking classifier model...
python -c "from utils import download_model; exit(0 if download_model(url='https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/model_classifier.pth', dest_path='models_classifier/model_classifier.pth') else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Error downloading classifier model. Please check your internet connection and try again.
    pause
    exit /b 1
)

:: Verify installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'[✓] PyTorch {torch.__version__}'); print(f'[✓] CUDA available: {torch.cuda.is_available()}'); print(f'[✓] GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
if %ERRORLEVEL% NEQ 0 (
    echo [!] Failed to verify PyTorch installation. Aborting.
    pause
    exit /b 1
)

:: Start the application
echo.
echo [*] Starting PyPotteryLens...
python app.py

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] An error occurred while starting PyPotteryLens. Please check the messages above.
    pause
    exit /b 1
)

:: Success - pause to keep window open
pause