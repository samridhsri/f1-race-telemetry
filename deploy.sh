#!/bin/bash

# F1 Race Flux Deployment Script for Linux/Mac
# This script stops existing containers, rebuilds images, and starts all services

echo "========================================"
echo "F1 Race Flux - Deployment Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

print_status "Stopping existing containers..."
docker-compose down

print_status "Removing old volumes (if any)..."
docker volume prune -f

print_status "Building Docker images..."
docker-compose build

if [ $? -ne 0 ]; then
    print_error "Failed to build Docker images."
    exit 1
fi

print_status "Starting all services..."
docker-compose up -d

if [ $? -ne 0 ]; then
    print_error "Failed to start services."
    exit 1
fi

print_status "Waiting for services to initialize (30 seconds)..."
sleep 30

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Services Status:"
docker-compose ps
echo ""
echo "Access Points:"
echo "----------------------------------------"
echo "  API Documentation:       http://localhost:8000/docs"
echo "  Driver Analysis:         http://localhost:8501"
echo "  Lap Times:               http://localhost:8502"
echo "  Position Tracking:       http://localhost:8503"
echo "  3D Driver Simulation:    http://localhost:8504"
echo "  Race Results:            http://localhost:8505"
echo "  Weather Data:            http://localhost:8506"
echo "  MLflow UI:               http://localhost:5001"
echo "----------------------------------------"
echo ""
echo "Useful Commands:"
echo "  View logs:               docker-compose logs -f [service-name]"
echo "  Stop services:           docker-compose down"
echo "  Restart services:        docker-compose restart"
echo "  View running containers: docker-compose ps"
echo ""
print_status "To fetch race data, navigate to http://localhost:8000/docs"
print_status "Use the /fetch_and_stream endpoint with year, event, and session_type"
echo ""
