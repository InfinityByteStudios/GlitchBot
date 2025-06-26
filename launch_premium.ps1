#!/usr/bin/env pwsh

# Premium AI Assistant Launcher
# Showcases the beautiful new interface inspired by Claude, ChatGPT, and Gemini

Clear-Host

# ASCII Art Header
Write-Host ""
Write-Host "██████╗ ██████╗ ███████╗███╗   ███╗██╗██╗   ██╗███╗   ███╗    ██████╗ ██╗" -ForegroundColor Cyan
Write-Host "██╔══██╗██╔══██╗██╔════╝████╗ ████║██║██║   ██║████╗ ████║   ██╔═══██╗██║" -ForegroundColor Cyan
Write-Host "██████╔╝██████╔╝█████╗  ██╔████╔██║██║██║   ██║██╔████╔██║   ██║   ██║██║" -ForegroundColor Blue
Write-Host "██╔═══╝ ██╔══██╗██╔══╝  ██║╚██╔╝██║██║██║   ██║██║╚██╔╝██║   ██║   ██║██║" -ForegroundColor Blue
Write-Host "██║     ██║  ██║███████╗██║ ╚═╝ ██║██║╚██████╔╝██║ ╚═╝ ██║   ╚██████╔╝██║" -ForegroundColor Magenta
Write-Host "╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═════╝ ╚═╝" -ForegroundColor Magenta
Write-Host ""
Write-Host "                    🚀 AI ASSISTANT PREMIUM EDITION 🚀" -ForegroundColor Yellow
Write-Host "          Inspired by Claude's thoughtfulness, ChatGPT's reliability," -ForegroundColor White
Write-Host "                    and Gemini's innovative design" -ForegroundColor White
Write-Host ""

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$premiumApp = Join-Path $PSScriptRoot "phase7_application\premium_ai_assistant.py"
$standardApp = Join-Path $PSScriptRoot "phase7_application\ai_assistant_app.py"

Write-Host "🌟 Choose Your Experience:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. 🎨 Premium Interface (RECOMMENDED)" -ForegroundColor Green
Write-Host "   • Beautiful gradients inspired by all three UIs" -ForegroundColor Gray
Write-Host "   • Advanced animations and hover effects" -ForegroundColor Gray
Write-Host "   • Premium avatars and enhanced personality" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 🚀 Standard Gradio Interface" -ForegroundColor Cyan
Write-Host "   • Clean, modern design" -ForegroundColor Gray
Write-Host "   • Fast and responsive" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 📱 Streamlit Interface" -ForegroundColor Blue
Write-Host "   • Professional dashboard style" -ForegroundColor Gray
Write-Host "   • Great for mobile devices" -ForegroundColor Gray
Write-Host ""
Write-Host "4. 💻 Command Line Interface" -ForegroundColor White
Write-Host "   • Direct terminal interaction" -ForegroundColor Gray
Write-Host "   • Perfect for developers" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🎨 Launching Premium AI Assistant..." -ForegroundColor Green
        Write-Host "✨ Loading beautiful interface with premium features..." -ForegroundColor Cyan
        Write-Host "🌐 Open your browser to: http://localhost:7860" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "🎭 Features included:" -ForegroundColor Magenta
        Write-Host "  • Claude-inspired thoughtful responses" -ForegroundColor White
        Write-Host "  • ChatGPT-style clean layout" -ForegroundColor White
        Write-Host "  • Gemini-inspired colorful gradients" -ForegroundColor White
        Write-Host "  • Enhanced personality and avatars" -ForegroundColor White
        Write-Host "  • Premium animations and effects" -ForegroundColor White
        Write-Host ""
        & $pythonPath $premiumApp --port 7860
    }
    "2" {
        Write-Host ""
        Write-Host "🚀 Launching Standard Gradio Interface..." -ForegroundColor Cyan
        Write-Host "🌐 Open your browser to: http://localhost:7860" -ForegroundColor Yellow
        Write-Host ""
        & $pythonPath $standardApp --interface gradio --port 7860
    }
    "3" {
        Write-Host ""
        Write-Host "📱 Launching Streamlit Interface..." -ForegroundColor Blue
        Write-Host "🌐 Will open automatically in your browser" -ForegroundColor Yellow
        Write-Host ""
        streamlit run $standardApp
    }
    "4" {
        Write-Host ""
        Write-Host "💻 Launching CLI Interface..." -ForegroundColor White
        Write-Host ""
        & $pythonPath $standardApp --interface cli
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice. Launching Premium Interface by default..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "🎨 Launching Premium AI Assistant..." -ForegroundColor Green
        Write-Host "✨ Loading beautiful interface with premium features..." -ForegroundColor Cyan
        Write-Host "🌐 Open your browser to: http://localhost:7860" -ForegroundColor Yellow
        Write-Host ""
        & $pythonPath $premiumApp --port 7860
    }
}

Write-Host ""
Write-Host "Thanks for using AI Assistant Premium! 🚀" -ForegroundColor Green
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
