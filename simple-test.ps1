# Simple API Test - Clean version without special characters
param(
    [string]$ApiUrl = "http://localhost:8001",
    [string]$ApiKey = "dev-key"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test Health
Write-Host "1. Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method GET
    Write-Host "   SUCCESS: Server is healthy!" -ForegroundColor Green
    Write-Host "   Model: $($health.model)" -ForegroundColor White
    Write-Host "   Threshold: $($health.threshold)" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "   ERROR: Server not responding" -ForegroundColor Red
    Write-Host "   Message: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure server is running: python serve_api.py --port 8001" -ForegroundColor Yellow
    exit 1
}

# Test Prediction
Write-Host "2. Testing Prediction Endpoint..." -ForegroundColor Yellow
$testEmail = "Click here to verify your account"
Write-Host "   Email: $testEmail" -ForegroundColor Gray
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

    Write-Host "   SUCCESS: Prediction complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "   Results:" -ForegroundColor Cyan
    Write-Host "   --------" -ForegroundColor Cyan
    $labelColor = if($result.is_phishing){"Red"}else{"Green"}
    Write-Host "   Label       : $($result.label)" -ForegroundColor $labelColor
    Write-Host "   Probability : $([math]::Round($result.probability * 100, 2))%" -ForegroundColor White
    Write-Host "   Confidence  : $($result.confidence)" -ForegroundColor White
    Write-Host "   Is Phishing : $($result.is_phishing)" -ForegroundColor White
    Write-Host "   Time (ms)   : $([math]::Round($result.response_time_ms, 2))" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "   ERROR: Prediction failed" -ForegroundColor Red
    Write-Host "   Message: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Test with different examples
Write-Host "3. Testing Multiple Examples..." -ForegroundColor Yellow
Write-Host ""

$examples = @(
    @{text="Thank you for the meeting notes"; expected="Safe"},
    @{text="URGENT! Your account will be suspended!"; expected="Phishing"},
    @{text="Meeting at 10 AM tomorrow"; expected="Safe"}
)

foreach ($example in $examples) {
    try {
        $headers = @{"x-api-key" = $ApiKey}
        $body = @{text = $example.text} | ConvertTo-Json

        $result = Invoke-RestMethod -Uri "$ApiUrl/predict" `
            -Method POST `
            -Headers $headers `
            -Body $body `
            -ContentType "application/json"

        $shortText = if($example.text.Length -gt 40) { $example.text.Substring(0, 37) + "..." } else { $example.text }
        $prob = [math]::Round($result.probability * 100, 1)
        $labelColor = if($result.is_phishing){"Red"}else{"Green"}

        Write-Host "   [$($result.label.PadRight(8))] $prob% - $shortText" -ForegroundColor $labelColor
    }
    catch {
        Write-Host "   [ERROR] Could not test: $($example.text)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All tests completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
