@echo off
REM Disco UI one-click installer for Windows.
REM
REM Python is INCLUDED in this package, in the python_embeded folder,
REM Python 3.10, ComfyUI portable style. There is nothing to install
REM first, no venv is created, and no Python on your system is
REM touched or needed -- if you have no Python at all, this still
REM works.
REM
REM What it does:
REM   1. Bootstraps pip into the bundled Python, first run only,
REM      using the bundled get-pip.py.
REM   2. Installs PyTorch with CUDA support, needs an NVIDIA GPU and
REM      a recent driver, and the rest of the requirements, into
REM      python_embeded\Lib\site-packages.
REM   3. Fetches the two open-source code folders the engine needs:
REM      OpenAI CLIP and the Disco-patched guided-diffusion fork.
REM      NOTE: guided-diffusion must come from kostarion's fork, NOT
REM      stock openai/guided-diffusion. The engine calls the sampler
REM      with Disco-only arguments, skip_timesteps, init_image,
REM      randomize_class, transformation_fn; stock openai rejects
REM      them with "unexpected keyword argument 'skip_timesteps'".
REM   4. Creates models\ and images_out\.
REM
REM What it does NOT do:
REM   - Download diffusion model weights. The app downloads the stock
REM     512 / 256 / secondary models automatically from their public
REM     sources on first run, see MODELS.md.
REM   - Bundle any portrait models. Those belong to their creators --
REM     see MODELS.md for how to add your own file.
REM
REM Requirements: Windows, an NVIDIA GPU. Everything here is free --
REM no account, no payment, no server.

setlocal
cd /d "%~dp0"

if exist python_embeded\python.exe goto HavePython
echo.
echo The bundled Python in python_embeded is missing.
echo Re-download or fully re-extract this package, then run
echo this file again.
echo.
pause
exit /b 1

:HavePython
echo Bundled Python:
python_embeded\python.exe --version

if exist python_embeded\Scripts\pip.exe goto HavePip
echo Setting up pip for the bundled Python - first run only...
python_embeded\python.exe python_embeded\get-pip.py --no-warn-script-location
if not errorlevel 1 goto HavePip
echo.
echo Pip setup failed. Check your internet connection,
echo then run this again.
echo.
pause
exit /b 1

:HavePip
python_embeded\python.exe -m pip install --no-warn-script-location --upgrade pip

echo Installing PyTorch with CUDA support - this is the big download...
python_embeded\python.exe -m pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto TorchOk
echo.
echo PyTorch install failed. Check your internet connection and
echo that your GPU driver is up to date, then run this again.
echo.
pause
exit /b 1

:TorchOk
echo Installing the remaining packages...
python_embeded\python.exe -m pip install --no-warn-script-location -r requirements.txt
if not errorlevel 1 goto ReqOk
echo.
echo Package install failed. Run this again -- pip resumes
echo downloads it already finished.
echo.
pause
exit /b 1

:ReqOk
where git >nul 2>&1
if errorlevel 1 goto NoGit

if exist CLIP goto GitClipOk
echo Fetching OpenAI CLIP...
git clone https://github.com/openai/CLIP.git CLIP
if exist CLIP goto GitClipOk
goto FailClip

:GitClipOk
if exist guided-diffusion goto Folders
echo Fetching Disco's guided-diffusion - kostarion fork...
git clone https://github.com/kostarion/guided-diffusion.git guided-diffusion
if exist guided-diffusion goto Folders
goto FailGd

:NoGit
echo Git not found -- downloading the code folders as ZIPs instead.
if exist CLIP goto ZipClipOk
echo Fetching OpenAI CLIP...
powershell -Command "Invoke-WebRequest -Uri https://github.com/openai/CLIP/archive/refs/heads/main.zip -OutFile CLIP.zip"
powershell -Command "Expand-Archive -Path CLIP.zip -DestinationPath . -Force"
move CLIP-main CLIP >nul
del CLIP.zip
if exist CLIP goto ZipClipOk
goto FailClip

:ZipClipOk
if exist guided-diffusion goto Folders
echo Fetching Disco's guided-diffusion - kostarion fork...
powershell -Command "Invoke-WebRequest -Uri https://github.com/kostarion/guided-diffusion/archive/refs/heads/main.zip -OutFile guided-diffusion.zip"
powershell -Command "Expand-Archive -Path guided-diffusion.zip -DestinationPath . -Force"
move guided-diffusion-main guided-diffusion >nul
del guided-diffusion.zip
if exist guided-diffusion goto Folders
goto FailGd

:FailClip
echo CLIP download failed. Check your internet connection,
echo then run this again.
pause
exit /b 1

:FailGd
echo guided-diffusion download failed. Check your internet
echo connection, then run this again.
pause
exit /b 1

:Folders
if not exist models mkdir models
if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start Disco UI.
echo.
pause
