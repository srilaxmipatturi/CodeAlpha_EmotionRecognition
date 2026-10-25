Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Speech Emotion Recognition Web Application" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m streamlit run app.py
} else {
    & "C:\Program Files\Python311\python.exe" -m streamlit run app.py
}
