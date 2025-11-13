#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import signal

print("Installing MLflow and dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow', 'psycopg2-binary'], check=True)

# Import MLflow AFTER installation
import mlflow.server

# Kill any existing processes on port 5000
print("Checking for existing processes on port 5000...")
try:
    subprocess.run(['pkill', '-f', 'gunicorn'], check=False)
    subprocess.run(['pkill', '-f', 'mlflow'], check=False)
    time.sleep(2)
    print("Cleaned up existing processes")
except:
    print("No existing processes to clean up")

# Set environment variables (this is how MLflow server reads configuration)
os.environ['MLFLOW_BACKEND_STORE_URI'] = 'postgresql://mlflow:mlflow@postgres:5432/mlflowdb'
os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = '/mlflow/artifacts'
os.environ['MLFLOW_SERVE_ARTIFACTS'] = 'true'

print("Starting MLflow server with environment variables...")

try:
    # Use the correct MLflow serve function with proper parameters
    mlflow.server.serve(
        host='0.0.0.0',
        port=5000,
        workers=4
    )
except Exception as e:
    print(f"MLflow serve function failed: {e}")
    print("Trying direct gunicorn approach...")
    
    # Alternative: use subprocess with gunicorn directly
    cmd = [
        'gunicorn',
        '--bind', '0.0.0.0:5000',
        '--workers', '4',
        '--timeout', '60',
        '--keep-alive', '2',
        '--max-requests', '1000',
        'mlflow.server:app'
    ]
    
    # Set MLflow environment variables
    env = os.environ.copy()
    env.update({
        'MLFLOW_BACKEND_STORE_URI': 'postgresql://mlflow:mlflow@postgres:5432/mlflowdb',
        'MLFLOW_DEFAULT_ARTIFACT_ROOT': '/mlflow/artifacts',
        'MLFLOW_SERVE_ARTIFACTS': 'true'
    })
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env) 