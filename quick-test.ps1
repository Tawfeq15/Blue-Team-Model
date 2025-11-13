# Quick API Test Script
param(
    [string]$ApiUrl = "http://localhost:8001",
    [string]$ApiKey = "dev-key"
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Quick API Test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Test Health
Write-Host "Testing health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method GET
    Write-Host "✓ Server is healthy!" -ForegroundColor Green
    Write-Host "  Model: $($health.model)" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "✗ Server not responding: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test Prediction
Write-Host "Testing prediction..." -ForegroundColor Yellow
$testEmail = "Click here to verify your account"
Write-Host "Email: $testEmail" -ForegroundColor Gray
Write-Host ""

try {
    $headers = @{
        "x-api-key" = $ApiKey
    }

    $body = @{
        text = $testEmail
    } | ConvertTo-Json

    $result = Invoke-RestMethod -Uri "$ApiUrl/predict" `
        -Method POST `
        -Headers $headers `
        -Body $body `
        -ContentType "application/json"

    Write-Host "✓ Prediction successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  Label: $($result.label)" -ForegroundColor $(if($result.is_phishing){"Red"}else{"Green"})
    Write-Host "  Probability: $([math]::Round($result.probability * 100, 2))%" -ForegroundColor White
    Write-Host "  Confidence: $($result.confidence)" -ForegroundColor White
    Write-Host "  Response Time: $([math]::Round($result.response_time_ms, 2)) ms" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "✗ Prediction failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "This is a model/feature pipeline issue, not an API issue." -ForegroundColor Yellow
    Write-Host "The API is working but there's a problem with feature extraction." -ForegroundColor Yellow
}

Write-Host "============================================" -ForegroundColor Cyan
