# AI-Assisted Fraud Detection - Setup Script
# Automates project setup and verification

$separator = "====================================================================="
Write-Host $separator -ForegroundColor Cyan
Write-Host " AI-ASSISTED FRAUD DETECTION - SETUP SCRIPT" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

# Step 1: Check Python installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
if (Test-Command python) {
    $pythonVersion = python --version
    Write-Host "  OK $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "  ERROR Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  OK Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "  ERROR Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate virtual environment and install dependencies
Write-Host ""
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Cyan

& ".\venv\Scripts\Activate.ps1"

# Upgrade pip first
python -m pip install --upgrade pip --quiet

# Install requirements
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK All dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  WARNING Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host "  Please check requirements.txt and install manually" -ForegroundColor Yellow
}

# Step 4: Verify installations
Write-Host ""
Write-Host "[4/6] Verifying installations..." -ForegroundColor Yellow

$packages = @(
    "cv2|opencv-python",
    "mediapipe|mediapipe",
    "deepface|deepface",
    "torch|pytorch",
    "sklearn|scikit-learn",
    "pandas|pandas",
    "numpy|numpy",
    "matplotlib|matplotlib",
    "streamlit|streamlit"
)

$allSuccess = $true
foreach ($package in $packages) {
    $import, $name = $package -split '\|'
    $result = python -c "import $import" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK $name" -ForegroundColor Green
    } else {
        Write-Host "  ERROR $name" -ForegroundColor Red
        $allSuccess = $false
    }
}

# Step 5: Create directory structure
Write-Host ""
Write-Host "[5/6] Setting up directory structure..." -ForegroundColor Yellow

$directories = @(
    "data\raw",
    "data\processed",
    "models",
    "logs"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Write-Host "  OK $dir" -ForegroundColor Green
}

# Step 6: Run quick test
Write-Host ""
Write-Host "[6/6] Running verification test..." -ForegroundColor Yellow

$testScript = @'
import sys
try:
    import cv2
    import mediapipe as mp
    import torch
    import sklearn
    import pandas as pd
    import numpy as np
    import streamlit as st
    from deepface import DeepFace
    print("SUCCESS")
    sys.exit(0)
except Exception as e:
    print("ERROR: " + str(e))
    sys.exit(1)
'@

$testResult = python -c $testScript 2>&1 | Out-String
if ($testResult -match "SUCCESS") {
    Write-Host "  OK All imports working correctly" -ForegroundColor Green
} else {
    Write-Host "  ERROR Some imports failed" -ForegroundColor Red
    Write-Host "  Details: $testResult" -ForegroundColor Gray
    $allSuccess = $false
}

# Final summary
Write-Host ""
Write-Host $separator -ForegroundColor Cyan
if ($allSuccess) {
    Write-Host " SETUP COMPLETE!" -ForegroundColor Green
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Generate training data:" -ForegroundColor White
    Write-Host "     python data_generator.py --samples 1000 --output data\synthetic_dataset.csv" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Train the model:" -ForegroundColor White
    Write-Host "     python train_model.py --data data\synthetic_dataset.csv" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Launch the web app:" -ForegroundColor White
    Write-Host "     streamlit run app.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "For more information, see README.md and QUICKSTART.md" -ForegroundColor Cyan
} else {
    Write-Host " SETUP INCOMPLETE" -ForegroundColor Yellow
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Some components failed to install." -ForegroundColor Yellow
    Write-Host "Please check the errors above and install missing packages manually." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Cyan
    Write-Host "  - Run: pip install mediapipe --no-cache-dir" -ForegroundColor Gray
    Write-Host "  - For GPU PyTorch: pip install torch torchvision" -ForegroundColor Gray
    Write-Host "  - See requirements.txt for all dependencies" -ForegroundColor Gray
}
Write-Host ""
