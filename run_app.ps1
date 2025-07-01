# Spotify User Clustering & Analysis App Launcher
Write-Host "Launching Spotify User Clustering & Analysis App..." -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Cyan
}
catch {
    Write-Host "Python is not installed or not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python and try again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if requirements are installed
Write-Host "Checking and installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run the Streamlit app
Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Green
streamlit run app.py

Read-Host "Press Enter to exit"
