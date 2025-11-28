# PowerShell script to run MAFIA training in Docker

param(
    [string]$HfToken = $env:HF_TOKEN,
    [string]$HfRepoId = "",
    [int]$HparamTrials = 10,
    [int]$HparamEpochs = 10,
    [int]$TrainEpochs = 100,
    [int]$NumSeeds = 5,
    [switch]$Build,
    [switch]$UploadAll,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host @"
MAFIA Docker Training Script
=============================

Usage:
  .\run_docker.ps1 [OPTIONS]

Options:
  -HfToken <token>          Hugging Face token (or use HF_TOKEN env variable)
  -HfRepoId <repo-id>       Hugging Face repository ID (e.g., 'username/model-name')
  -HparamTrials <int>       Number of Optuna trials (default: 10)
  -HparamEpochs <int>       Epochs per hparam trial (default: 10)
  -TrainEpochs <int>        Epochs for full training (default: 100)
  -NumSeeds <int>           Number of seeds to run (default: 5)
  -Build                    Rebuild Docker image before running
  -UploadAll                Upload all seed runs (not just official)
  -Help                     Show this help message

Examples:
  # Basic run without upload
  .\run_docker.ps1

  # Run with Hugging Face upload (official only)
  .\run_docker.ps1 -HfRepoId "yourusername/mafia-model"

  # Run with custom parameters
  .\run_docker.ps1 -HparamTrials 5 -TrainEpochs 50 -NumSeeds 3

  # Rebuild image and run
  .\run_docker.ps1 -Build -HfRepoId "yourusername/mafia-model"

  # Upload all seed runs (not just official)
  .\run_docker.ps1 -HfRepoId "yourusername/mafia-model" -UploadAll

"@
    exit 0
}

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "MAFIA Docker Training Pipeline" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Create necessary directories
$directories = @("data", "res", "log")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✓ Created directory: $dir" -ForegroundColor Green
    }
}

# Build Docker image if requested
if ($Build) {
    Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
    docker build -t mafia-rl-training:latest .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Docker build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Docker image built successfully" -ForegroundColor Green
}

# Prepare command
$cmd = "python scripts/auto_pipeline.py"
$cmd += " --hparam-trials $HparamTrials"
$cmd += " --hparam-epochs $HparamEpochs"
$cmd += " --train-epochs $TrainEpochs"
$cmd += " --num-seeds $NumSeeds"

# Add Hugging Face parameters if provided
if ($HfRepoId) {
    $cmd += " --hf-repo-id `"$HfRepoId`""
    if (-not $UploadAll) {
        $cmd += " --hf-upload-official-only"
    }
    Write-Host "✓ Hugging Face upload enabled: $HfRepoId" -ForegroundColor Green
}

# Prepare environment variables
$envVars = @()
if ($HfToken) {
    $envVars += "-e", "HF_TOKEN=$HfToken"
    Write-Host "✓ Hugging Face token configured" -ForegroundColor Green
} elseif ($HfRepoId) {
    Write-Host "⚠ Warning: HF_REPO_ID set but no HF_TOKEN provided" -ForegroundColor Yellow
    Write-Host "  Set HF_TOKEN environment variable or use -HfToken parameter" -ForegroundColor Yellow
}

# Print configuration
Write-Host "`nTraining Configuration:" -ForegroundColor Cyan
Write-Host "  Hparam Trials: $HparamTrials"
Write-Host "  Hparam Epochs: $HparamEpochs"
Write-Host "  Train Epochs: $TrainEpochs"
Write-Host "  Num Seeds: $NumSeeds"
if ($HfRepoId) {
    Write-Host "  HF Repository: $HfRepoId"
    Write-Host "  Upload Mode: $(if ($UploadAll) { 'All runs' } else { 'Official only' })"
}

Write-Host "`nStarting Docker container..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop training`n" -ForegroundColor Yellow

# Run Docker container
$dockerArgs = @(
    "run",
    "--rm",
    "-it",
    "-v", "${PWD}/data:/app/data",
    "-v", "${PWD}/res:/app/res",
    "-v", "${PWD}/log:/app/log"
)

# Add environment variables
$dockerArgs += $envVars

# Add image and command
$dockerArgs += "mafia-rl-training:latest"
$dockerArgs += "bash", "-c", $cmd

# Execute
& docker @dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Training completed successfully!" -ForegroundColor Green
    Write-Host "Results saved in: ./res/" -ForegroundColor Cyan
    Write-Host "Logs saved in: ./log/" -ForegroundColor Cyan
} else {
    Write-Host "`n✗ Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
