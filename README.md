# F1 Race Flux

## Overview
This repository develops a real-time Formula 1 data analytics system that processes live telemetry data through a streaming pipeline and presents interactive dashboards for race analysis, driver performance comparison, and ML-powered race predictions.

## Team

- Samridh Srivastava (ss18906)
- Hardik Setia (hs5595)
- Bryce Miranda (bm3986)

## Report and Presentation link

* **Report**: [View Report](https://drive.google.com/file/d/1T01AzpU4vXa4vlYIEUEZoE_7dLAKRqzw/view?usp=sharing)
* **Presentation**: [View Presentation](https://docs.google.com/presentation/d/18s8zbc-00wfGvrGvgqM6sVNclCIL7nIg/edit?usp=drive_link&ouid=111586551984895934380&rtpof=true&sd=true)


## Repository Structure

### `api/`
FastAPI backend for data fetching and REST endpoints.

### `consumer/`
Apache Spark streaming processors for real-time data transformation:
* Stream Processor: `f1_streaming_processor.py`
* Utilities: `kafka_spark/safe_utils.py`

### `producer/`
Data fetching logic from FastF1 API:
* Main Fetcher: `fetch_f1_data.py`

### `streamlit/`
Interactive dashboard applications:
* Driver Analysis: `driver_analysis.py`
* Lap Times: `lap_times.py`
* Position Tracking: `position_tracking.py`
* 3D Simulation: `driver_simulation.py`
* Race Results: `race_results.py`
* Weather Data: `weather_data.py`
* ML Predictions: `race_prediction_model.py`

### `mlflow/`
Machine learning experiment tracking configuration.

## Technologies
Utilizes Docker, Apache Kafka, Apache Spark, MongoDB, PostgreSQL, FastAPI, Streamlit, and MLflow. Python libraries include FastF1, Pandas, NumPy, Scikit-learn, and PySpark.

## Getting Started
To set up this project locally, run the following commands:

```bash
git clone https://github.com/samridhsri/f1-race-telemetry
cd f1-race-flux
```

Ensure Docker and Docker Compose are installed with at least 8GB RAM available.

## Starting the System

**Windows:**
```bash
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

This will start all 8 Docker services (Kafka, Spark, MongoDB, MLflow, API, Streamlit dashboards).

## Accessing the Applications

Once all services are running:

* **API Documentation**: http://localhost:8000/docs
* **Driver Analysis Dashboard**: http://localhost:8501
* **Lap Times Dashboard**: http://localhost:8502
* **Position Tracking Dashboard**: http://localhost:8503
* **3D Driver Simulation**: http://localhost:8504
* **Race Results Dashboard**: http://localhost:8505
* **Weather Data Dashboard**: http://localhost:8506
* **MLflow UI**: http://localhost:5001

## Running the Pipeline

1. **Fetch Race Data**: Navigate to http://localhost:8000/docs and use the `/fetch_and_stream` endpoint:
```json
{
  "year": 2024,
  "event": "Monaco",
  "session_type": "Race"
}
```

2. **View Real-Time Processing**: Data flows through Kafka → Spark → MongoDB automatically. Monitor progress in Docker logs:
```bash
docker-compose logs -f spark-processor
```

3. **Analyze in Dashboards**: Open any Streamlit dashboard (ports 8501-8506) to visualize the processed data.

4. **Run ML Predictions**: 
   * Open the main dashboard at http://localhost:8501
   * Navigate to "Race Predictions" tab
   * Select target race (e.g., 2025 Bahrain) and training years (e.g., 2022-2024)
   * Click "Run Prediction Model" to generate race outcome predictions

5. **Track Experiments**: View ML experiments, model versions, and metrics at http://localhost:5001

## Architecture Flow

```
FastF1 API → FastAPI → Kafka (6 topics) → Spark Streaming → MongoDB (6 collections) → Streamlit Dashboards
                                                                      ↓
                                                               MLflow Tracking
```

## Troubleshooting

**Services won't start:**
```bash
docker-compose down
docker-compose up -d
docker-compose logs [service-name]
```

**No data in dashboards:**
* Ensure you've fetched data via API first
* Check Kafka topics: `docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092`
* Verify MongoDB data: `docker exec -it mongodb mongosh`

**Memory issues:**
* Increase Docker Desktop memory allocation (Settings → Resources)
* Reduce memory limits in `docker-compose.yml`