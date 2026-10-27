@echo off
REM Start Disco UI. Run install.bat once first.
REM
REM The app treats this folder as the Disco install: models\ and
REM images_out\ live here, and the CLIP / guided-diffusion code folders
REM sit beside them. That is why DISCO_INSTALL_DIR points here.
REM
REM The app runs on the Python bundled in python_embeded\ -- no
REM venv, no system Python involved.

setlocal
cd /d "%~dp0"
set DISCO_INSTALL_DIR=%~dp0
if not exist python_embeded\python.exe (
    echo The bundled Python is missing. Re-extract the package,
    echo then run install.bat.
    pause
    exit /b 1
)
if not exist python_embeded\Lib\site-packages\gradio (
    echo The packages are not installed yet. Run install.bat first.
    pause
    exit /b 1
)
python_embeded\python.exe disco-ui\app.py
echo.
echo The app stopped. Double-click run.bat to start it again.
pause
