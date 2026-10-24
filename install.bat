@echo off
REM Disco UI one-click installer (Windows).
REM
REM What it does:
REM   1. Creates a Python virtual environment (venv) in this folder.
REM   2. Installs PyTorch with CUDA support (needs an NVIDIA GPU + a
REM      recent driver) and the rest of the requirements.
REM   3. Fetches the two open-source code folders the engine needs:
REM      OpenAI CLIP and the Disco-patched guided-diffusion fork.
REM      NOTE: guided-diffusion must come from kostarion's fork, NOT
REM      stock openai/guided-diffusion. The engine calls the sampler
REM      with Disco-only arguments (skip_timesteps, init_image,
REM      randomize_class, transformation_fn); stock openai rejects
REM      them with "unexpected keyword argument 'skip_timesteps'".
REM   4. Creates models\ and images_out\.
REM
REM What it does NOT do:
REM   - Download diffusion model weights. The app downloads the stock
REM     512 / 256 / secondary models automatically from their public
REM     sources on first run (see MODELS.md).
REM   - Bundle any portrait models. Those belong to their creators --
REM     see MODELS.md for how to add your own file.
REM
REM Requirements: Windows, Python 3.10 or newer on your PATH, an NVIDIA
REM GPU. Everything here is free -- no account, no payment, no server.

setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.10 or newer from
    echo python.org, tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support (this is the big download)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo PyTorch install failed. Check your internet connection and
    echo that your GPU driver is up to date, then run this again.
    pause
    exit /b 1
)

echo Installing the remaining packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed. Run this again -- pip resumes
    echo downloads it already finished.
    pause
    exit /b 1
)

where git >nul 2>&1
if errorlevel 1 goto :nogit

if not exist CLIP (
    echo Fetching OpenAI CLIP...
    git clone https://github.com/openai/CLIP.git CLIP
    if not exist CLIP (
        echo CLIP download failed. Check your internet connection,
        echo then run this again.
        pause
        exit /b 1
    )
)
if not exist guided-diffusion (
    echo Fetching Disco's guided-diffusion (kostarion fork)...
    git clone https://github.com/kostarion/guided-diffusion.git guided-diffusion
    if not exist guided-diffusion (
        echo guided-diffusion download failed. Check your internet
        echo connection, then run this again.
        pause
        exit /b 1
    )
)
goto :folders

:nogit
echo Git not found -- downloading the code folders as ZIPs instead.
if not exist CLIP (
    echo Fetching OpenAI CLIP...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/openai/CLIP/archive/refs/heads/main.zip -OutFile CLIP.zip"
    powershell -Command "Expand-Archive -Path CLIP.zip -DestinationPath . -Force"
    move CLIP-main CLIP >nul
    del CLIP.zip
    if not exist CLIP (
        echo CLIP download failed. Check your internet connection,
        echo then run this again.
        pause
        exit /b 1
    )
)
if not exist guided-diffusion (
    echo Fetching Disco's guided-diffusion (kostarion fork)...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/kostarion/guided-diffusion/archive/refs/heads/main.zip -OutFile guided-diffusion.zip"
    powershell -Command "Expand-Archive -Path guided-diffusion.zip -DestinationPath . -Force"
    move guided-diffusion-main guided-diffusion >nul
    del guided-diffusion.zip
    if not exist guided-diffusion (
        echo guided-diffusion download failed. Check your internet
        echo connection, then run this again.
        pause
        exit /b 1
    )
)

:folders
if not exist models mkdir models
if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start Disco UI.
echo.
pause
