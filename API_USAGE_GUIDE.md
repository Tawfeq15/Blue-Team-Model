# 🛡️ Phishing Detection API - Usage Guide

## Quick Start

### Start the Server
```powershell
python serve_api.py
```

The API will be available at: `http://localhost:8000`
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Default API Key:** `dev-key`

---

## PowerShell API Requests

### ⚠️ Common Issue: curl in PowerShell

**Problem:** In PowerShell, `curl` is an alias for `Invoke-WebRequest`, which has different syntax than Unix curl.

**This WON'T work:**
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"text": "test"}'
```

**Error:**
```
Cannot bind parameter 'Headers'. Cannot convert the "Content-Type: application/json"
value of type "System.String" to type "System.Collections.IDictionary".
```

---

## ✅ Solutions

### Method 1: Use the Test Script (Easiest!)

```powershell
# Run the test script
.\test-api.ps1

# Test with specific email
.\test-api.ps1 -TestEmail "Click here to verify your account"

# Use different API URL
.\test-api.ps1 -ApiUrl "http://192.168.1.100:8000"
```

The script demonstrates 3 different methods for making API calls!

---

### Method 2: Invoke-RestMethod (Recommended!)

```powershell
# Setup headers
$headers = @{
    "x-api-key" = "dev-key"
}

# Setup body
$body = @{
    text = "URGENT! Verify your account now!"
} | ConvertTo-Json

# Send request
$result = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method POST `
    -Headers $headers `
    -Body $body `
    -ContentType "application/json"

# Display results
Write-Host "Label: $($result.label)"
Write-Host "Probability: $([math]::Round($result.probability * 100, 2))%"
Write-Host "Is Phishing: $($result.is_phishing)"
Write-Host "Confidence: $($result.confidence)"
```

---

### Method 3: Invoke-WebRequest (Detailed)

```powershell
$headers = @{
    "Content-Type" = "application/json"
    "x-api-key" = "dev-key"
}

$body = @{
    text = "Click here to claim your prize!"
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
    -Method POST `
    -Headers $headers `
    -Body $body

$result = $response.Content | ConvertFrom-Json

Write-Host "Label: $($result.label)" -ForegroundColor $(if($result.is_phishing){"Red"}else{"Green"})
Write-Host "Confidence: $($result.confidence)"
Write-Host "Response Time: $([math]::Round($result.response_time_ms, 2)) ms"
```

---

### Method 4: Use Real curl.exe

```powershell
# Use curl.exe instead of the PowerShell alias
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -H "x-api-key: dev-key" `
  -d '{\"text\": \"Click here to verify your account\"}'
```

---

## API Endpoints

### POST /predict
Predict if a single email is phishing

**Request:**
```powershell
$headers = @{"x-api-key" = "dev-key"}
$body = @{text = "Your email text here"} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method POST -Headers $headers -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "text": "Your email text...",
  "probability": 0.95,
  "label": "Phishing",
  "is_phishing": true,
  "confidence": "Very High",
  "threshold": 0.5,
  "response_time_ms": 45.2
}
```

---

### POST /predict/batch
Predict multiple emails at once

**Request:**
```powershell
$headers = @{"x-api-key" = "dev-key"}
$body = @{
    texts = @(
        "Meeting at 10 AM tomorrow",
        "URGENT! Click here NOW!",
        "Thanks for the update"
    )
} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" `
    -Method POST -Headers $headers -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "total": 3,
  "phishing_count": 1,
  "safe_count": 2,
  "results": [
    {"text": "Meeting at 10 AM...", "probability": 0.1, "label": "Safe", "is_phishing": false},
    {"text": "URGENT! Click here...", "probability": 0.95, "label": "Phishing", "is_phishing": true},
    {"text": "Thanks for the update", "probability": 0.05, "label": "Safe", "is_phishing": false}
  ],
  "response_time_ms": 120.5
}
```

---

### POST /explain
Get explanation for why an email was classified as phishing/safe

**Request:**
```powershell
$headers = @{"x-api-key" = "dev-key"}
$body = @{text = "URGENT! Verify your account now!"} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:8000/explain" `
    -Method POST -Headers $headers -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "text": "URGENT! Verify your account now!",
  "probability": 0.95,
  "is_phishing": true,
  "explanation": {
    "available": true,
    "method": "feature_importance",
    "top_positive": [
      ["urgent", 0.45],
      ["verify", 0.32],
      ["account", 0.28]
    ],
    "top_negative": [
      ["thank", -0.15],
      ["regards", -0.12]
    ]
  }
}
```

---

### GET /health
Check API health (no API key required)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "model": "CalibratedClassifierCV",
  "threshold": 0.5,
  "artifacts_loaded": {
    "model": true,
    "pipeline": true,
    "cleaner": false,
    "drift_monitor": false
  },
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.92,
    "f1": 0.925,
    "roc_auc": 0.98
  }
}
```

---

### GET /stats
Get API usage statistics (no API key required)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/stats"
```

**Response:**
```json
{
  "total_predictions": 1250,
  "phishing_detected": 450,
  "safe_emails": 800,
  "start_time": "2025-11-13T10:00:00",
  "avg_response_time_ms": 42.5,
  "uptime_seconds": 7200,
  "phishing_rate": 36.0
}
```

---

## Reusable PowerShell Function

Create a reusable function for testing:

```powershell
function Test-Phishing {
    param(
        [string]$EmailText,
        [string]$ApiUrl = "http://localhost:8000",
        [string]$ApiKey = "dev-key"
    )

    $headers = @{"x-api-key" = $ApiKey}
    $body = @{text = $EmailText} | ConvertTo-Json

    $result = Invoke-RestMethod -Uri "$ApiUrl/predict" `
        -Method POST `
        -Headers $headers `
        -Body $body `
        -ContentType "application/json"

    # Display results with colors
    $color = if($result.is_phishing){"Red"}else{"Green"}
    Write-Host "🛡️ Result: $($result.label)" -ForegroundColor $color
    Write-Host "   Probability: $([math]::Round($result.probability * 100, 2))%"
    Write-Host "   Confidence: $($result.confidence)"

    return $result
}

# Usage examples
Test-Phishing "Click here to claim your prize!"
Test-Phishing "Meeting at 10 AM tomorrow"
Test-Phishing "URGENT! Verify your account now!"
```

---

## Common Issues & Solutions

### Issue 1: "Invalid API key"

**Problem:** Missing or incorrect API key header

**Solution:**
```powershell
# Make sure to include the correct header
$headers = @{
    "x-api-key" = "dev-key"  # ✓ Correct header name
}
```

---

### Issue 2: "Cannot bind parameter 'Headers'"

**Problem:** Using PowerShell's curl alias incorrectly

**Solution:**
```powershell
# ❌ Wrong - using curl alias
curl -H "Content-Type: application/json"

# ✓ Correct - use Invoke-RestMethod
Invoke-RestMethod -Headers @{"Content-Type"="application/json"}

# ✓ Or use real curl.exe
curl.exe -H "Content-Type: application/json"
```

---

### Issue 3: Server not responding

**Problem:** API server is not running

**Solution:**
```powershell
# Start the server
python serve_api.py

# Check if port 8000 is in use
netstat -ano | findstr :8000

# Test health endpoint
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

---

### Issue 4: Connection refused

**Problem:** Firewall or network issue

**Solution:**
```powershell
# Try with 127.0.0.1 instead of localhost
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"

# Check Windows Firewall settings
# Allow Python through the firewall if needed
```

---

## Security Notes

1. **API Key:** The default key `dev-key` is for development only. Change it in production:
   ```powershell
   $env:API_KEY = "your-secure-key-here"
   python serve_api.py
   ```

2. **HTTPS:** For production, use HTTPS instead of HTTP

3. **Rate Limiting:** Consider implementing rate limiting for production use

4. **Access Control:** Restrict API access by IP or use more sophisticated authentication

---

## Examples for Different Languages

### Python
```python
import requests

headers = {"x-api-key": "dev-key"}
data = {"text": "URGENT! Verify your account now!"}

response = requests.post("http://localhost:8000/predict",
                        json=data,
                        headers=headers)

result = response.json()
print(f"Label: {result['label']}")
print(f"Probability: {result['probability']:.2%}")
```

### JavaScript (Node.js)
```javascript
const fetch = require('node-fetch');

async function testPhishing(text) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': 'dev-key'
        },
        body: JSON.stringify({ text })
    });

    const result = await response.json();
    console.log(`Label: ${result.label}`);
    console.log(`Probability: ${(result.probability * 100).toFixed(2)}%`);
}

testPhishing("URGENT! Click here now!");
```

### cURL (Bash/Linux)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"text": "URGENT! Verify your account now!"}'
```

---

## Performance Tips

1. **Batch Requests:** Use `/predict/batch` for multiple emails instead of making separate requests
2. **Keep-Alive:** Reuse HTTP connections when making multiple requests
3. **Async Requests:** Use async/await patterns for concurrent requests
4. **Caching:** Cache results for identical emails if applicable

---

## Need Help?

- Check API documentation: http://localhost:8000/docs
- Run health check: http://localhost:8000/health
- Use the test script: `.\test-api.ps1`
- Check logs in the server terminal

---

**Happy Phishing Detection! 🛡️**
