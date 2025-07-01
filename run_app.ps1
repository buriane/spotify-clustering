# Aplikasi Analisis dan Clustering Pengguna Spotify Launcher
Write-Host "Meluncurkan Aplikasi Analisis dan Clustering Pengguna Spotify..." -ForegroundColor Green
Write-Host ""

# Check if Python is installed (versi 3.13 direkomendasikan)
try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Cyan
    
    # Verifikasi versi Python
    if (-not ($pythonVersion -like "*3.13*")) {
        Write-Host "Peringatan: Aplikasi ini dioptimalkan untuk Python 3.13" -ForegroundColor Yellow
    }
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
