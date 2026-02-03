# GFN Test Runner
# Usage: .\run_tests.ps1 [test_file]

param(
    [string]$TestFile = "tests/unit/"
)

Write-Host "Running GFN Tests..." -ForegroundColor Cyan
Write-Host "Test target: $TestFile" -ForegroundColor Yellow
Write-Host ""

# Set PYTHONPATH to project root
$env:PYTHONPATH = $PSScriptRoot

# Run pytest
python -m pytest $TestFile -v --tb=short

Write-Host ""
Write-Host "Tests completed!" -ForegroundColor Green
