#!/usr/bin/env pwsh

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   🚀 AI Assistant - Beautiful UI Launcher" -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $PSScriptRoot "phase7_application\ai_assistant_app.py"

Write-Host "Choose your interface:" -ForegroundColor Yellow
Write-Host "1. Gradio Web Interface (Recommended)" -ForegroundColor Green
Write-Host "2. Streamlit Interface" -ForegroundColor Green
Write-Host "3. Command Line Interface" -ForegroundColor Green
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 Starting beautiful Gradio interface..." -ForegroundColor Green
        Write-Host "🌐 Open your browser to: http://localhost:7860" -ForegroundColor Cyan
        Write-Host ""
        & $pythonPath $appPath --interface gradio --port 7860
    }
    "2" {
        Write-Host ""
        Write-Host "🚀 Starting Streamlit interface..." -ForegroundColor Green
        Write-Host "🌐 Will open automatically in your browser" -ForegroundColor Cyan
        Write-Host ""
        streamlit run $appPath
    }
    "3" {
        Write-Host ""
        Write-Host "🚀 Starting CLI interface..." -ForegroundColor Green
        Write-Host ""
        & $pythonPath $appPath --interface cli
    }
    default {
        Write-Host "Invalid choice. Defaulting to Gradio interface..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "🚀 Starting beautiful Gradio interface..." -ForegroundColor Green
        Write-Host "🌐 Open your browser to: http://localhost:7860" -ForegroundColor Cyan
        Write-Host ""
        & $pythonPath $appPath --interface gradio --port 7860
    }
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
