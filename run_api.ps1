# run_api.ps1
Write-Host "Starting Blue Team Phishing API Server..." -ForegroundColor Cyan
Write-Host "=" -repeat 60 -ForegroundColor Gray

$env:API_KEY = "dev-key"
Write-Host "API_KEY set to: dev-key" -ForegroundColor Green

Write-Host ""
Write-Host "Server will run on: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "API Documentation: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "API Key required: x-api-key: dev-key" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "=" -repeat 60 -ForegroundColor Gray
Write-Host ""

uvicorn serve_api:app --host 127.0.0.1 --port 8000 --reload
