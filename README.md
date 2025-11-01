# F1 RaceFlux - Formula 1 Data Pipeline

A comprehensive real-time Formula 1 data processing pipeline with advanced ML lifecycle management, analytics, and prediction capabilities. This project collects, processes, and visualizes F1 telemetry data using a modern data engineering and MLOps stack with MLflow integration.

## How to Run

1. Ensure Docker and Docker Compose are installed
2. Start all services: `docker-compose up -d`
3. Access the applications:
   - Streamlit Dashboard: http://localhost:8501
   - MLflow Tracking UI: http://localhost:5001
   - API Documentation: http://localhost:8000/docs

## What It Does

F1 RaceFlux is a complete data pipeline that captures real-time Formula 1 race telemetry data, processes it using Apache Spark, stores it in MongoDB, and provides RESTful API access. The system includes an interactive Streamlit dashboard for data visualization and race analytics, along with MLflow integration for machine learning model training, experiment tracking, and race outcome predictions. The pipeline uses Kafka for real-time data streaming, enabling scalable processing of F1 telemetry data with advanced analytics including driver performance comparison, track speed heatmaps, lap time analysis, and race predictions using gradient boosting models.
