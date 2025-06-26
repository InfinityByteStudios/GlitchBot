@echo off
echo.
echo ============================================
echo   🚀 AI Assistant - Beautiful UI Launcher
echo ============================================
echo.

set PYTHON_PATH=%~dp0.venv\Scripts\python.exe
set APP_PATH=%~dp0phase7_application\ai_assistant_app.py

echo Choose your interface:
echo 1. Gradio Web Interface (Recommended)
echo 2. Streamlit Interface  
echo 3. Command Line Interface
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting beautiful Gradio interface...
    echo 🌐 Open your browser to: http://localhost:7860
    echo.
    "%PYTHON_PATH%" "%APP_PATH%" --interface gradio --port 7860
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Streamlit interface...
    echo 🌐 Will open automatically in your browser
    echo.
    streamlit run "%APP_PATH%"
) else if "%choice%"=="3" (
    echo.
    echo 🚀 Starting CLI interface...
    echo.
    "%PYTHON_PATH%" "%APP_PATH%" --interface cli
) else (
    echo Invalid choice. Defaulting to Gradio interface...
    echo.
    echo 🚀 Starting beautiful Gradio interface...
    echo 🌐 Open your browser to: http://localhost:7860
    echo.
    "%PYTHON_PATH%" "%APP_PATH%" --interface gradio --port 7860
)

pause
