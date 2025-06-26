@echo off
title Ultimate AI Assistant Launcher
color 0B

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                               ║
echo ║        ✨ ULTIMATE AI ASSISTANT LAUNCHER ✨                  ║
echo ║                                                               ║
echo ║        🎨 Pixel-Perfect UI Design                            ║
echo ║        🚀 Premium User Experience                            ║
echo ║        💎 Claude + ChatGPT + Gemini Inspired                ║
echo ║                                                               ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    echo    Download from: https://python.org/downloads
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

echo 🔍 Checking required packages...
echo 🔧 Installing/updating dependencies...
python -m pip install --upgrade gradio torch transformers streamlit >nul 2>&1
if errorlevel 1 (
    echo ❌ Failed to install packages. Please run as administrator.
    pause
    exit /b 1
)

echo ✅ All dependencies ready!
echo.

echo 🎨 Available AI Assistant Interfaces:
echo    1. ✨ Ultimate AI Assistant (Recommended)
echo    2. 💎 Premium AI Assistant  
echo    3. 🔥 Ultra Premium Assistant
echo    4. 📱 Standard AI Assistant
echo    5. 🚪 Exit
echo.

set /p choice="👉 Select interface (1-5): "

if "%choice%"=="1" (
    echo 🎯 Launching Ultimate AI Assistant...
    echo 🔄 Initializing premium interface...
    timeout /t 1 >nul
    
    if exist "phase7_application\ultimate_ai_assistant.py" (
        cd phase7_application
        python ultimate_ai_assistant.py
    ) else (
        echo ❌ Ultimate AI Assistant not found!
        echo    Expected: phase7_application\ultimate_ai_assistant.py
    )
) else if "%choice%"=="2" (
    echo 🎯 Launching Premium AI Assistant...
    echo 🔄 Initializing premium interface...
    timeout /t 1 >nul
    
    if exist "phase7_application\premium_ai_assistant.py" (
        cd phase7_application
        python premium_ai_assistant.py
    ) else (
        echo ❌ Premium AI Assistant not found!
    )
) else if "%choice%"=="3" (
    echo 🎯 Launching Ultra Premium Assistant...
    echo 🔄 Initializing ultra premium interface...
    timeout /t 1 >nul
    
    if exist "phase7_application\ultra_premium_assistant.py" (
        cd phase7_application
        python ultra_premium_assistant.py
    ) else (
        echo ❌ Ultra Premium Assistant not found!
    )
) else if "%choice%"=="4" (
    echo 🎯 Launching Standard AI Assistant...
    echo 🔄 Initializing standard interface...
    timeout /t 1 >nul
    
    if exist "phase7_application\ai_assistant_app.py" (
        cd phase7_application
        python ai_assistant_app.py
    ) else (
        echo ❌ Standard AI Assistant not found!
    )
) else if "%choice%"=="5" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo ❌ Invalid selection. Please choose 1-5.
)

echo.
echo 🎉 Thank you for using Ultimate AI Assistant!
echo 💡 Tip: Bookmark this launcher for easy access!
pause
