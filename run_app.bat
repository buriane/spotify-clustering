@echo off
echo Meluncurkan Aplikasi Analisis dan Clustering Pengguna Spotify...
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not found in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking and installing requirements...
pip install -r requirements.txt

REM Run the Streamlit app
echo.
echo Starting Streamlit app...
streamlit run app.py

pause
