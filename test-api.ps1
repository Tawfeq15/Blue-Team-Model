# ============================================================
# 🛡️ Phishing Detection API - PowerShell Test Script
# ============================================================
# This script demonstrates how to properly call the API from PowerShell

param(
    [string]$ApiUrl = "http://localhost:8000",
    [string]$ApiKey = "dev-key",
    [string]$TestEmail = ""
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🛡️  Phishing Detection API - PowerShell Client" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Example emails
$examples = @{
    "safe" = "Thank you for the update. I'll review the document and get back to you by tomorrow. Best regards."
    "phishing" = "URGENT! Your account has been compromised! Click here immediately: http://suspicious-site.com/verify"
    "test" = "Click here to verify your account: http://login-secure-check.com/verify"
}

# If no email provided, use the test example
if ([string]::IsNullOrEmpty($TestEmail)) {
    Write-Host "No email provided. Using test example..." -ForegroundColor Yellow
    $TestEmail = $examples["test"]
    Write-Host "Email text: $TestEmail" -ForegroundColor Gray
    Write-Host ""
}

# ====================
# Method 1: Using Invoke-WebRequest (PowerShell native)
# ====================
function Test-PhishingInvokeWebRequest {
    param([string]$Text, [string]$Url, [string]$Key)

    Write-Host "Method 1: Using Invoke-WebRequest..." -ForegroundColor Green

    try {
        $headers = @{
            "Content-Type" = "application/json"
            "x-api-key" = $Key
        }

        $body = @{
            text = $Text
        } | ConvertTo-Json

        $response = Invoke-WebRequest -Uri "$Url/predict" `
            -Method POST `
            -Headers $headers `
            -Body $body `
            -ContentType "application/json"

        $result = $response.Content | ConvertFrom-Json

        Write-Host "✓ Success!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Results:" -ForegroundColor Cyan
        Write-Host "  Label: $($result.label)" -ForegroundColor $(if($result.is_phishing){"Red"}else{"Green"})
        Write-Host "  Probability: $([math]::Round($result.probability * 100, 2))%" -ForegroundColor White
        Write-Host "  Confidence: $($result.confidence)" -ForegroundColor White
        Write-Host "  Response Time: $([math]::Round($result.response_time_ms, 2)) ms" -ForegroundColor White
        Write-Host ""

        return $result
    }
    catch {
        Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        return $null
    }
}

# ====================
# Method 2: Using Invoke-RestMethod (Simpler)
# ====================
function Test-PhishingInvokeRestMethod {
    param([string]$Text, [string]$Url, [string]$Key)

    Write-Host "Method 2: Using Invoke-RestMethod..." -ForegroundColor Green

    try {
        $headers = @{
            "x-api-key" = $Key
        }

        $body = @{
            text = $Text
        } | ConvertTo-Json

        $result = Invoke-RestMethod -Uri "$Url/predict" `
            -Method POST `
            -Headers $headers `
            -Body $body `
            -ContentType "application/json"

        Write-Host "✓ Success!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Results:" -ForegroundColor Cyan
        Write-Host "  Label: $($result.label)" -ForegroundColor $(if($result.is_phishing){"Red"}else{"Green"})
        Write-Host "  Probability: $([math]::Round($result.probability * 100, 2))%" -ForegroundColor White
        Write-Host "  Confidence: $($result.confidence)" -ForegroundColor White
        Write-Host "  Response Time: $([math]::Round($result.response_time_ms, 2)) ms" -ForegroundColor White
        Write-Host ""

        return $result
    }
    catch {
        Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        return $null
    }
}

# ====================
# Method 3: Using curl.exe (Real curl, not PowerShell alias)
# ====================
function Test-PhishingCurl {
    param([string]$Text, [string]$Url, [string]$Key)

    Write-Host "Method 3: Using curl.exe (real curl)..." -ForegroundColor Green

    try {
        # Create JSON body
        $jsonBody = @{
            text = $Text
        } | ConvertTo-Json -Compress

        # Escape quotes for curl
        $escapedBody = $jsonBody -replace '"', '\"'

        # Use curl.exe explicitly (not the PowerShell alias)
        $result = curl.exe -X POST "$Url/predict" `
            -H "Content-Type: application/json" `
            -H "x-api-key: $Key" `
            -d $jsonBody `
            --silent

        if ($LASTEXITCODE -eq 0) {
            $parsed = $result | ConvertFrom-Json

            Write-Host "✓ Success!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Results:" -ForegroundColor Cyan
            Write-Host "  Label: $($parsed.label)" -ForegroundColor $(if($parsed.is_phishing){"Red"}else{"Green"})
            Write-Host "  Probability: $([math]::Round($parsed.probability * 100, 2))%" -ForegroundColor White
            Write-Host "  Confidence: $($parsed.confidence)" -ForegroundColor White
            Write-Host "  Response Time: $([math]::Round($parsed.response_time_ms, 2)) ms" -ForegroundColor White
            Write-Host ""

            return $parsed
        }
        else {
            Write-Host "✗ curl.exe failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host ""
            return $null
        }
    }
    catch {
        Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Note: curl.exe might not be available on your system" -ForegroundColor Yellow
        Write-Host ""
        return $null
    }
}

# ====================
# Test Health Endpoint (No API key required)
# ====================
function Test-Health {
    param([string]$Url)

    Write-Host "Testing health endpoint..." -ForegroundColor Cyan

    try {
        $health = Invoke-RestMethod -Uri "$Url/health" -Method GET

        Write-Host "✓ API is healthy!" -ForegroundColor Green
        Write-Host "  Model: $($health.model)" -ForegroundColor White
        Write-Host "  Threshold: $($health.threshold)" -ForegroundColor White
        Write-Host "  Uptime: $([math]::Round($health.uptime_seconds, 2)) seconds" -ForegroundColor White
        Write-Host ""

        return $true
    }
    catch {
        Write-Host "✗ API is not responding: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Make sure the server is running with: python serve_api.py" -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
}

# ====================
# Main Execution
# ====================

# Test health first
if (-not (Test-Health -Url $ApiUrl)) {
    Write-Host "Exiting due to API unavailability" -ForegroundColor Red
    exit 1
}

# Run all three methods
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Testing API with sample email..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Test-PhishingInvokeWebRequest -Text $TestEmail -Url $ApiUrl -Key $ApiKey
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

Test-PhishingInvokeRestMethod -Text $TestEmail -Url $ApiUrl -Key $ApiKey
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

Test-PhishingCurl -Text $TestEmail -Url $ApiUrl -Key $ApiKey
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "💡 Quick Examples:" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "# Test with a safe email:" -ForegroundColor Yellow
Write-Host '.\test-api.ps1 -TestEmail "Meeting tomorrow at 10am"' -ForegroundColor White
Write-Host ""
Write-Host "# Test with a phishing email:" -ForegroundColor Yellow
Write-Host '.\test-api.ps1 -TestEmail "URGENT! Click here now!"' -ForegroundColor White
Write-Host ""
Write-Host "# Use different API URL:" -ForegroundColor Yellow
Write-Host '.\test-api.ps1 -ApiUrl "http://192.168.1.100:8000"' -ForegroundColor White
Write-Host ""
Write-Host "# Use different API key:" -ForegroundColor Yellow
Write-Host '.\test-api.ps1 -ApiKey "your-custom-key"' -ForegroundColor White
Write-Host ""
