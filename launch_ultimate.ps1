# Ultimate AI Assistant Launcher
# PowerShell script for launching the premium AI assistant interface

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║        ✨ ULTIMATE AI ASSISTANT LAUNCHER ✨                  ║" -ForegroundColor Yellow
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║        🎨 Pixel-Perfect UI Design                            ║" -ForegroundColor Green
Write-Host "║        🚀 Premium User Experience                            ║" -ForegroundColor Green
Write-Host "║        💎 Claude + ChatGPT + Gemini Inspired                ║" -ForegroundColor Green
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host ""
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    Write-Host "   Download from: https://python.org/downloads" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "🔍 Checking required packages..." -ForegroundColor Yellow

# Check and install required packages
$requiredPackages = @("gradio", "torch", "transformers", "streamlit")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        python -c "import $package" 2>$null
        Write-Host "✅ $package - OK" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $package - Missing" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "🔧 Installing missing packages..." -ForegroundColor Yellow
    Write-Host "   Packages: $($missingPackages -join ', ')" -ForegroundColor Cyan
    
    try {
        python -m pip install $missingPackages
        Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install packages. Please run as administrator." -ForegroundColor Red
        pause
        exit 1
    }
}

Write-Host ""
Write-Host "🎨 Available AI Assistant Interfaces:" -ForegroundColor Cyan
Write-Host "   1. 🎯 Pixel Perfect Assistant (Exact UI Recreation)" -ForegroundColor Yellow
Write-Host "   2. 💎 Premium AI Assistant" -ForegroundColor Magenta
Write-Host "   3. 🔥 Ultra Premium Assistant" -ForegroundColor Red
Write-Host "   4. ✨ Ultimate AI Assistant" -ForegroundColor Blue
Write-Host "   5. 📱 Standard AI Assistant" -ForegroundColor Green
Write-Host "   6. 🚪 Exit" -ForegroundColor Gray

Write-Host ""
$choice = Read-Host "👉 Select interface (1-6)"

switch ($choice) {
    "1" {
        Write-Host "🎯 Launching Pixel Perfect AI Assistant..." -ForegroundColor Green
        Write-Host "🔄 Initializing pixel-perfect interface..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
        
        if (Test-Path "phase7_application\pixel_perfect_assistant.py") {
            Set-Location "phase7_application"
            & python pixel_perfect_assistant.py
        } else {
            Write-Host "❌ Pixel Perfect AI Assistant not found!" -ForegroundColor Red
            Write-Host "   Expected: phase7_application\pixel_perfect_assistant.py" -ForegroundColor Yellow
        }
    }
    "2" {
        Write-Host "🎯 Launching Premium AI Assistant..." -ForegroundColor Magenta
        Write-Host "🔄 Initializing premium interface..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
        
        if (Test-Path "phase7_application\premium_ai_assistant.py") {
            Set-Location "phase7_application"
            & python premium_ai_assistant.py
        } else {
            Write-Host "❌ Premium AI Assistant not found!" -ForegroundColor Red
        }
    }
    "3" {
        Write-Host "🎯 Launching Ultra Premium Assistant..." -ForegroundColor Red
        Write-Host "🔄 Initializing ultra premium interface..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
        
        if (Test-Path "phase7_application\ultra_premium_assistant.py") {
            Set-Location "phase7_application"
            & python ultra_premium_assistant.py
        } else {
            Write-Host "❌ Ultra Premium Assistant not found!" -ForegroundColor Red
        }
    }
    "4" {
        Write-Host "🎯 Launching Ultimate AI Assistant..." -ForegroundColor Blue
        Write-Host "🔄 Initializing ultimate interface..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
        
        if (Test-Path "phase7_application\ultimate_ai_assistant.py") {
            Set-Location "phase7_application"
            & python ultimate_ai_assistant.py
        } else {
            Write-Host "❌ Ultimate AI Assistant not found!" -ForegroundColor Red
        }
    }
    "5" {
        Write-Host "🎯 Launching Standard AI Assistant..." -ForegroundColor Green
        Write-Host "🔄 Initializing standard interface..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
        
        if (Test-Path "phase7_application\ai_assistant_app.py") {
            Set-Location "phase7_application"
            & python ai_assistant_app.py
        } else {
            Write-Host "❌ Standard AI Assistant not found!" -ForegroundColor Red
        }
    }
    "6" {
        Write-Host "👋 Goodbye!" -ForegroundColor Green
        exit 0
    }
    default {
        Write-Host "❌ Invalid selection. Please choose 1-6." -ForegroundColor Red
        pause
    }
}

Write-Host ""
Write-Host "🎉 Thank you for using Ultimate AI Assistant!" -ForegroundColor Green
Write-Host "💡 Tip: Bookmark this launcher for easy access!" -ForegroundColor Cyan
pause
