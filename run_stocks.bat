@echo off
:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to initialize...
    :loop
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        timeout /t 2 >nul
        goto loop
    )
)

echo Docker is ready! Starting your Stock App...
cd C:\Users\sar81\stocks
docker compose up -d

:: Open the browser immediately
echo Launching Stock Forecast App...
start http://localhost:8501

:: Keep the terminal open to show logs
docker compose logs -f