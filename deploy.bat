@echo off
REM F1 Race Flux Deployment Script for Windows
REM This script stops existing containers, rebuilds images, and starts all services

echo ========================================
echo F1 Race Flux - Deployment Script
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [INFO] Stopping existing containers...
docker-compose down

echo [INFO] Removing old volumes (if any)...
docker volume prune -f

echo [INFO] Building Docker images...
docker-compose build

if %errorlevel% neq 0 (
    echo [ERROR] Failed to build Docker images.
    pause
    exit /b 1
)

echo [INFO] Starting all services...
docker-compose up -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services.
    pause
    exit /b 1
)

echo [INFO] Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo Services Status:
docker-compose ps
echo.
echo Access Points:
echo ----------------------------------------
echo   API Documentation:       http://localhost:8000/docs
echo   Driver Analysis:         http://localhost:8501
echo   Lap Times:               http://localhost:8502
echo   Position Tracking:       http://localhost:8503
echo   3D Driver Simulation:    http://localhost:8504
echo   Race Results:            http://localhost:8505
echo   Weather Data:            http://localhost:8506
echo   MLflow UI:               http://localhost:5001
echo ----------------------------------------
echo.
echo Useful Commands:
echo   View logs:               docker-compose logs -f [service-name]
echo   Stop services:           docker-compose down
echo   Restart services:        docker-compose restart
echo   View running containers: docker-compose ps
echo.
echo [INFO] To fetch race data, navigate to http://localhost:8000/docs
echo [INFO] Use the /fetch_and_stream endpoint with year, event, and session_type
echo.
pause
