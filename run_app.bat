@echo off
echo ===================================================
echo Starting Speech Emotion Recognition Web Application
echo ===================================================
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    py -3.11 -m streamlit run app.py
)
pause
