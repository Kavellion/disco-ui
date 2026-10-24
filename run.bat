@echo off
REM Start Disco UI. Run install.bat once first.
REM
REM The app treats this folder as the Disco install: models\ and
REM images_out\ live here, and the CLIP / guided-diffusion code folders
REM sit beside them. That is why DISCO_INSTALL_DIR points here.

setlocal
cd /d "%~dp0"
set DISCO_INSTALL_DIR=%~dp0
if not exist venv\Scripts\activate.bat (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python disco-ui\app.py
echo.
echo The app stopped. Double-click run.bat to start it again.
pause
