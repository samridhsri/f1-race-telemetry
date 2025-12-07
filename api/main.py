from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fastf1
from fastf1.ergast import Ergast
import os
import logging
import pymongo
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
import docker

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("F1API")
# FastF1 Cache
cache_dir = "/app/f1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
# MongoDB connection
client = pymongo.MongoClient("mongodb://mongodb:27017/")
db = client["f1db"]
collection = db["telemetry"]
circuit_collection = db["circuits"]
# Docker client
docker_client = docker.from_env()

def get_docker_network_name():
    """Get the Docker network name by finding the network that contains required services"""
    try:
        # First, try to get network from current container (api container) - most reliable
        try:
            api_container = docker_client.containers.get("api")
            if api_container.attrs.get("NetworkSettings", {}).get("Networks"):
                network_name = list(api_container.attrs["NetworkSettings"]["Networks"].keys())[0]
                logger.info(f"Detected network from API container: {network_name}")
                return network_name
        except Exception as e:
            logger.warning(f"Could not get network from API container: {e}")
        
        # Fallback: Look for a network that contains kafka and mongodb containers
        try:
            networks = docker_client.networks.list()
            for network in networks:
                try:
                    # Refresh network to get containers
                    network.reload()
                    containers = network.containers
                    container_names = [c.name for c in containers]
                    if "kafka" in container_names and "mongodb" in container_names:
                        logger.info(f"Detected network from container search: {network.name}")
                        return network.name
                except Exception as e:
                    logger.warning(f"Error checking network {network.name}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error listing networks: {e}")
        
        # Try to find network by common patterns
        try:
            networks = docker_client.networks.list()
            for network in networks:
                # Look for networks with common docker-compose naming patterns
                if "_default" in network.name and network.name != "bridge":
                    logger.info(f"Using network with default pattern: {network.name}")
                    return network.name
        except:
            pass
        
        # Final fallback - try common names
        fallback_names = ["f1-race-flux_default", "f1-data-pipeline_default"]
        for name in fallback_names:
            try:
                network = docker_client.networks.get(name)
                logger.info(f"Using fallback network: {name}")
                return name
            except:
                continue
        
        # Last resort fallback
        logger.error("Could not detect network name, using hardcoded fallback")
        return "f1-race-flux_default"
    except Exception as e:
        logger.error(f"Could not detect network name: {e}, using fallback")
        return "f1-race-flux_default"

# Network name will be detected lazily when needed
# Don't cache it at module load time as containers may not be ready

# FastAPI setup
app = FastAPI(title="F1 Data API")
# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class Race(BaseModel):
    year: int
    name: str
    round: int
    date: str


class SessionInfo(BaseModel):
    name: str
    key: str
    date: Optional[str] = None


class EventSessions(BaseModel):
    year: int
    event: str
    sessions: List[SessionInfo]


class RaceDataRequest(BaseModel):
    year: int
    event: str
    session: str


class PredictionRequest(BaseModel):
    event: str
    years: str = "2022,2023,2024"
    force_reprocess: bool = False


@app.get("/")
async def root():
    return {"message": "F1 Data API is running"}


@app.get("/available-races", response_model=List[Race])
async def get_available_races():
    races = []
    for year in range(2022, 2027):  # Include 2022-2026 for historical data and future predictions
        try:
            schedule = fastf1.get_event_schedule(year)
            for _, event in schedule.iterrows():
                races.append(
                    {
                        "year": year,
                        "name": event["EventName"],
                        "round": event["RoundNumber"],
                        "date": event["EventDate"].strftime("%Y-%m-%d"),
                    }
                )
        except Exception as e:
            logger.error(f"Failed to fetch schedule for {year}: {e}")
    return races


@app.get("/sessions/{year}/{event}", response_model=EventSessions)
async def get_sessions(year: int, event: str):
    try:
        if event.startswith("Pre-Season"):
            logger.info(
                f"Pre-Season event requested: {event}. Returning empty sessions."
            )
            return {"year": year, "event": event, "sessions": []}
        schedule = fastf1.get_event_schedule(year)
        event_info = None
        for _, row in schedule.iterrows():
            if event.lower() in row["EventName"].lower():
                event_info = row
                break
        if event_info is None:
            try:
                round_number = int(event)
                for _, row in schedule.iterrows():
                    if row["RoundNumber"] == round_number:
                        event_info = row
                        break
            except ValueError:
                pass
        if event_info is None:
            raise HTTPException(status_code=404, detail="Event not found")
        round_number = event_info["RoundNumber"]
        event_data = fastf1.get_event(year, round_number)
        session_map = {
            "FP1": "Practice 1",
            "FP2": "Practice 2",
            "FP3": "Practice 3",
            "Q": "Qualifying",
            "S": "Sprint",
            "SQ": "Sprint Qualifying",
            "R": "Race",
        }
        sessions = []
        for key, name in session_map.items():
            try:
                date = event_data.get_session_date(key)
                if date:
                    sessions.append(
                        {"name": name, "key": key, "date": date.strftime("%Y-%m-%d")}
                    )
            except Exception:
                pass
        return {"year": year, "event": event_info["EventName"], "sessions": sessions}
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail="Error fetching sessions")


@app.post("/fetch-race-data")
async def fetch_race_data(request: RaceDataRequest):
    try:
        logger.info(
            f"Requested fetch: {request.year} {request.event} {request.session}"
        )
        schedule = fastf1.get_event_schedule(request.year)
        event_info = None
        for _, row in schedule.iterrows():
            if request.event.lower() in row["EventName"].lower():
                event_info = row.to_dict()
                break
        if event_info is None:
            try:
                round_number = int(request.event)
                for _, row in schedule.iterrows():
                    if row["RoundNumber"] == round_number:
                        event_info = row.to_dict()
                        break
            except ValueError:
                pass
        if event_info is None:
            raise HTTPException(status_code=404, detail="Event not found")

        event_param = event_info["EventName"]

        # Check if data exists in all required collections
        required_collections = [
            "car_position",
            "car_telemetry",
            "driver_info",
            "race_results",
            "telemetry",
            "weather",
        ]

        collection_counts = {}
        all_collections_have_data = True

        # Build the query to check for existing data
        session_map = {
            "FP1": "Practice 1",
            "FP2": "Practice 2",
            "FP3": "Practice 3",
            "Q": "Qualifying",
            "S": "Sprint",
            "SQ": "Sprint Qualifying",
            "R": "Race",
        }
        query = {
            "Year": request.year,
            "GrandPrix": event_param,
            "SessionType": session_map.get(request.session.upper(), request.session),
        }

        # Check each collection for data matching our query
        for coll_name in required_collections:
            try:
                collection = db[coll_name]
                count = collection.count_documents(query)
                collection_counts[coll_name] = count
                logger.info(
                    f"Found {count} records in {coll_name} collection for this session"
                )

                # If any collection has zero documents, mark all_collections_have_data as False
                if count == 0:
                    all_collections_have_data = False
            except Exception as e:
                logger.error(f"Error checking collection {coll_name}: {e}")
                all_collections_have_data = False
                collection_counts[coll_name] = 0

        # If we have data in all collections, skip the producer
        if all_collections_have_data:
            logger.info(
                f"Found existing data for this session in all collections — skipping producer."
            )
            return {
                "status": "exists",
                "message": "Data already exists in all required collections",
                "counts": collection_counts,
            }

        # Check if a producer container for this session is already running
        container_filter = {
            "ancestor": "producer-image",
            "label": [
                f"year={request.year}",
                f"event={event_param}",
                f"session={request.session}",
            ],
        }

        running_containers = docker_client.containers.list(filters=container_filter)

        if running_containers:
            logger.info(
                f"Producer already running for this session in container {running_containers[0].short_id}"
            )
            return {
                "status": "already_running",
                "container_id": running_containers[0].short_id,
                "message": "Producer is already running for this session",
                "counts": collection_counts,
            }

        # If we don't have data in all collections, run the producer
        missing_collections = [
            coll for coll, count in collection_counts.items() if count == 0
        ]
        logger.info(f"Missing data in collections: {missing_collections}")

        container = docker_client.containers.run(
            image="producer-image",
            command=[
                "python",
                "f1_data_producer.py",
                "--year",
                str(request.year),
                "--event",
                event_param,
                "--session",
                request.session,
            ],
            environment={
                "KAFKA_BROKER": "kafka:29092",
                "MONGO_URI": "mongodb://mongodb:27017/f1db",
            },
            volumes={"/tmp/checkpoint": {"bind": "/tmp/checkpoint", "mode": "rw"}},
            network=get_docker_network_name(),
            labels={
                "year": str(request.year),
                "event": event_param,
                "session": request.session,
            },
            detach=True,
            remove=False,
        )

        logger.info(f"Started container {container.short_id}")
        return {
            "status": "started",
            "container_id": container.short_id,
            "message": f"Processing data for missing collections: {missing_collections}",
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except docker.errors.DockerException as e:
        logger.error(f"Docker error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Docker error: {str(e)}")
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection error: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to fetch race data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/check-processing-status")
async def check_processing_status(request: RaceDataRequest):
    """
    Check if data processing is complete by looking for data in MongoDB collections
    """
    try:
        logger.info(
            f"Checking processing status: {request.year} {request.event} {request.session}"
        )

        schedule = fastf1.get_event_schedule(request.year)
        event_info = None
        for _, row in schedule.iterrows():
            if request.event.lower() in row["EventName"].lower():
                event_info = row.to_dict()
                break
        if event_info is None:
            try:
                round_number = int(request.event)
                for _, row in schedule.iterrows():
                    if row["RoundNumber"] == round_number:
                        event_info = row.to_dict()
                        break
            except ValueError:
                pass
        if event_info is None:
            raise HTTPException(status_code=404, detail="Event not found")

        event_param = event_info["EventName"]

        # Get the session type mapping
        session_map = {
            "FP1": "Practice 1",
            "FP2": "Practice 2",
            "FP3": "Practice 3",
            "Q": "Qualifying",
            "S": "Sprint",
            "SQ": "Sprint Qualifying",
            "R": "Race",
        }

        # Build the query
        query = {
            "Year": request.year,
            "GrandPrix": event_param,
            "SessionType": session_map.get(request.session.upper(), request.session),
        }

        # Check if data exists in all required collections
        required_collections = [
            "car_position",
            "car_telemetry",
            "driver_info",
            "race_results",
            "telemetry",
            "weather",
        ]

        collection_counts = {}
        all_collections_have_data = True
        total_data_count = 0

        # Check each collection for data matching our query
        for coll_name in required_collections:
            try:
                collection = db[coll_name]
                count = collection.count_documents(query)
                collection_counts[coll_name] = count
                total_data_count += count

                # If any collection has zero documents, mark all_collections_have_data as False
                if count == 0:
                    all_collections_have_data = False
            except Exception as e:
                logger.error(f"Error checking collection {coll_name}: {e}")
                all_collections_have_data = False
                collection_counts[coll_name] = 0

        # Check container status first to determine if processing is truly complete
        container_status = "stopped"
        container_id = None
        container_exited_successfully = False
        
        try:
            all_producer_containers = docker_client.containers.list(
                filters={"ancestor": "producer-image"}, all=True
            )
            
            for container in all_producer_containers:
                labels = container.labels
                container_year = labels.get("year")
                container_event = labels.get("event")
                container_session = labels.get("session")
                
                if (
                    container_year == str(request.year)
                    and container_event == event_param
                    and container_session == request.session
                ):
                    container.reload()
                    status = container.status.lower()
                    
                    if "running" in status or "up" in status:
                        container_status = "running"
                        container_id = container.short_id
                        break
                    elif "exited" in status:
                        exit_code = container.attrs.get("State", {}).get("ExitCode", -1)
                        if exit_code == 0:
                            container_exited_successfully = True
                            container_id = container.short_id
                            logger.info(
                                f"Container {container.short_id} finished successfully with {total_data_count} total records"
                            )
                        else:
                            logger.warning(
                                f"Container {container.short_id} exited with error code {exit_code}"
                            )
        except Exception as e:
            logger.error(f"Error checking container status: {e}")

        # If all collections have data, processing is complete
        if all_collections_have_data:
            return {
                "status": "complete",
                "message": "Data processing complete - all collections have data",
                "container_status": container_status,
                "container_id": container_id,
                "counts": collection_counts,
            }
        # If container exited successfully and we have some data, consider it complete
        elif container_exited_successfully and total_data_count > 0:
            return {
                "status": "complete",
                "message": f"Data processing complete - container finished successfully with {total_data_count} total records",
                "container_status": "stopped",
                "container_id": container_id,
                "counts": collection_counts,
            }
        # If container is still running, it's processing
        elif container_status == "running":
            return {
                "status": "processing",
                "message": f"Data processing in progress. Container is running. Found {total_data_count} records so far.",
                "container_status": container_status,
                "container_id": container_id,
                "counts": collection_counts,
            }
        else:
            # Container stopped but no data or incomplete data
            if container_exited_successfully:
                # Container finished but data might still be processing in Spark/consumer
                # Or data might be incomplete
                return {
                    "status": "processing",
                    "message": f"Container finished, but data may still be processing. Found {total_data_count} records so far. Some collections may still be empty.",
                    "container_status": "stopped",
                    "container_id": container_id,
                    "counts": collection_counts,
                }
            else:
                # No container found or container failed
                return {
                    "status": "processing",
                    "message": f"No active container found. Found {total_data_count} records. Data may still be processing or container may have failed.",
                    "container_status": container_status,
                    "container_id": container_id,
                    "counts": collection_counts,
                }

    except Exception as e:
        logger.error(f"Failed to check processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/circuit-data/{year}")
async def get_circuit_data(year: int):
    """
    Fetches circuit data for the specified year from Ergast API
    and stores it in MongoDB for use by the analytics service.
    """
    try:
        logger.info(f"Fetching circuit data for year {year} from Ergast API")

        # Check if we already have circuit data for this year in MongoDB
        existing_data = list(circuit_collection.find({"year": year}, {"_id": 0}))
        if existing_data:
            logger.info(f"Using cached circuit data for {year}")
            return {"year": year, "circuits": existing_data}

        # If not in database, fetch from Ergast API
        ergast = Ergast()
        circuits = ergast.get_circuits(season=year, result_type="raw")

        # Store in MongoDB and format for response
        circuit_data = []
        for circuit in circuits:
            # Format the circuit data
            circuit_info = {
                "year": year,
                "circuitId": circuit["circuitId"],
                "circuitName": circuit["circuitName"],
                "url": circuit["url"],
                "latitude": float(circuit["Location"]["lat"]),
                "longitude": float(circuit["Location"]["long"]),
                "locality": circuit["Location"]["locality"],
                "country": circuit["Location"]["country"],
            }

            # Store in MongoDB
            circuit_collection.insert_one(circuit_info)
            circuit_data.append(circuit_info)

        logger.info(f"Stored {len(circuit_data)} circuit records for {year}")
        return {"year": year, "circuits": circuit_data}

    except Exception as e:
        logger.error(f"Error fetching circuit data for {year}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching circuit data: {str(e)}"
        )


@app.get("/event-schedule/{year}")
async def get_event_schedule_with_locations(year: int):
    """
    Fetches F1 race schedule for the specified year and merges with circuit location data.
    """
    try:
        logger.info(f"Fetching schedule for year {year}")

        # Get the event schedule from fastf1
        schedule = fastf1.get_event_schedule(year)

        # Fetch circuit data (this will use cached data if available)
        circuit_response = await get_circuit_data(year)
        circuits = circuit_response["circuits"]

        # Create a mapping of circuit IDs to their coordinates
        circuit_mapping = {}
        locality_mapping = {}
        country_mapping = {}

        for circuit in circuits:
            # Create multiple mappings to increase match chances
            circuit_mapping[circuit["circuitName"]] = circuit
            circuit_mapping[circuit["circuitId"]] = circuit

            # Map by locality (city)
            locality_mapping[circuit["locality"].lower()] = circuit

            # Map by country with locality
            country_key = f"{circuit['locality']}, {circuit['country']}".lower()
            country_mapping[country_key] = circuit

        # Convert schedule to list of dictionaries with location data
        events = []
        for _, event in schedule.iterrows():
            # Basic event info
            location = {
                "round": int(event["RoundNumber"]),
                "name": event["EventName"],
                "country": event["Country"],
                "location": event["Location"],
                "date": event["EventDate"].strftime("%Y-%m-%d"),
                "lat": None,
                "long": None,
                "locality": None,
                "circuitId": None,
            }

            # Try to find this circuit in our circuit mapping
            # Method 1: Try direct match by circuit name
            matched_circuit = None
            for circuit_name, circuit in circuit_mapping.items():
                if (
                    event["Location"].lower() in circuit_name.lower()
                    or circuit_name.lower() in event["Location"].lower()
                ):
                    matched_circuit = circuit
                    break

            # Method 2: Try match by city/locality
            if not matched_circuit:
                if event["Location"].lower() in locality_mapping:
                    matched_circuit = locality_mapping[event["Location"].lower()]

            # Method 3: Try match by city, country combination
            if not matched_circuit:
                location_key = f"{event['Location']}, {event['Country']}".lower()
                if location_key in country_mapping:
                    matched_circuit = country_mapping[location_key]

            # If we found a match, use its coordinates
            if matched_circuit:
                location["lat"] = matched_circuit["latitude"]
                location["long"] = matched_circuit["longitude"]
                location["locality"] = matched_circuit["locality"]
                location["circuitId"] = matched_circuit["circuitId"]

            events.append(location)

        return {"year": year, "events": events}

    except Exception as e:
        logger.error(f"Error fetching schedule for {year}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching schedule: {str(e)}"
        )


@app.get("/prediction-events")
async def get_prediction_events():
    """
    Get available events for prediction (tries 2026, falls back to 2025 if not available)
    """
    # Try 2026 first, then fall back to 2025 if 2026 schedule is not available yet
    years_to_try = [2026, 2025]

    for year in years_to_try:
        try:
            logger.info(f"Attempting to fetch {year} race schedule for predictions")
            schedule = fastf1.get_event_schedule(year)

            # Check if we got valid data
            if schedule is None or len(schedule) == 0:
                logger.warning(f"{year} schedule is empty, trying next year")
                continue

            events = []
            for _, event in schedule.iterrows():
                events.append(
                    {
                        "year": year,
                        "name": event["EventName"],
                        "round": event["RoundNumber"],
                        "date": event["EventDate"].strftime("%Y-%m-%d"),
                        "location": event["Location"],
                        "country": event["Country"],
                    }
                )

            logger.info(f"Successfully fetched {len(events)} events from {year} schedule")
            return events

        except Exception as e:
            logger.warning(f"Failed to fetch {year} schedule: {e}")
            if year == years_to_try[-1]:
                # This was the last year to try
                logger.error(f"All attempts failed. Last error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not fetch race schedule. 2026 schedule may not be available yet in FastF1. Try using 2025 data for now. Error: {str(e)}"
                )
            continue


@app.post("/start-prediction")
async def start_prediction(request: PredictionRequest):
    """
    Start a prediction process for a specified 2025 event
    using historical data from 2022-2024
    """
    try:
        logger.info(
            f"Starting prediction for event: {request.event} with years: {request.years}"
        )

        # Get the event schedule for 2025
        schedule = fastf1.get_event_schedule(2025)
        event_info = None

        # Find the matching event
        for _, row in schedule.iterrows():
            if request.event.lower() in row["EventName"].lower():
                event_info = row
                break

        if event_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Event '{request.event}' not found in 2025 schedule",
            )

        event_name = event_info["EventName"]
        logger.info(f"Found matching 2025 event: {event_name}")

        # Check if data already exists
        query = {"Year": 2025, "GrandPrix": event_name, "IsPrediction": True}

        # Check if data exists and we're not forcing reprocessing
        if not request.force_reprocess:
            count = collection.count_documents(query)
            if count > 0:
                logger.info(
                    f"Prediction data already exists for {event_name} with {count} records"
                )
                return {
                    "status": "exists",
                    "message": f"Prediction data already exists for {event_name}",
                    "count": count,
                }

        # Start the producer container with prediction mode
        container = docker_client.containers.run(
            image="producer-image",
            command=[
                "python",
                "f1_data_producer.py",
                "--year",
                "2025",
                "--event",
                event_name,
                "--session",
                "R",
                "--prediction-mode",
                "--source-years",
                request.years,
            ],
            environment={
                "KAFKA_BROKER": "kafka:29092",
                "MONGO_URI": "mongodb://mongodb:27017/f1db",
            },
            volumes={"/tmp/checkpoint": {"bind": "/tmp/checkpoint", "mode": "rw"}},
            network=get_docker_network_name(),
            labels={"type": "prediction_processor", "prediction_event": event_name},
            detach=True,
            remove=False,
        )

        logger.info(f"Started prediction container {container.short_id}")
        return {
            "status": "started",
            "container_id": container.short_id,
            "message": f"Prediction process started for {event_name}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start prediction process: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-prediction-status")
async def check_prediction_status(request: PredictionRequest):
    """
    Check the status of a prediction process
    """
    try:
        logger.info(f"Checking prediction status for: {request.event}")

        # Get the event schedule for 2025
        schedule = fastf1.get_event_schedule(2025)
        event_info = None

        # Find the matching event
        for _, row in schedule.iterrows():
            if request.event.lower() in row["EventName"].lower():
                event_info = row
                break

        if event_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Event '{request.event}' not found in 2025 schedule",
            )

        event_name = event_info["EventName"]

        # Check MongoDB for prediction data
        query = {"Year": 2025, "GrandPrix": event_name, "IsPrediction": True}

        # Check each collection for prediction data
        collection_counts = {}
        for coll_name in [
            "telemetry",
            "car_telemetry",
            "car_position",
            "driver_info",
            "race_results",
            "weather",
        ]:
            try:
                coll = db[coll_name]
                count = coll.count_documents(query)
                collection_counts[coll_name] = count
            except Exception as e:
                logger.error(f"Error checking collection {coll_name}: {e}")
                collection_counts[coll_name] = 0

        # Check if any data exists
        total_count = sum(collection_counts.values())

        # Check if a prediction container is still running
        container_filter = {
            "ancestor": "producer-image",
            "label": ["type=prediction_processor", f"prediction_event={event_name}"],
        }

        running_containers = docker_client.containers.list(filters=container_filter)
        container_status = "stopped"
        container_id = None

        if running_containers:
            container_status = "running"
            container_id = running_containers[0].short_id

        # Determine status based on data and container
        if total_count > 0:
            if container_status == "running":
                status = "processing"
                message = f"Prediction data is being processed. Found {total_count} records so far."
            else:
                status = "complete"
                message = (
                    f"Prediction data processing complete. Found {total_count} records."
                )
        else:
            if container_status == "running":
                status = "starting"
                message = "Prediction process is starting. No data available yet."
            else:
                status = "not_found"
                message = "No prediction data found and no process is running."

        return {
            "status": status,
            "message": message,
            "container_status": container_status,
            "container_id": container_id,
            "counts": collection_counts,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check prediction status: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prediction-results/{event}")
async def get_prediction_results(event: str):
    """
    Get prediction results for a specified event
    """
    try:
        # Find the matching 2025 event
        schedule = fastf1.get_event_schedule(2025)
        event_info = None

        for _, row in schedule.iterrows():
            if event.lower() in row["EventName"].lower():
                event_info = row
                break

        if event_info is None:
            raise HTTPException(
                status_code=404, detail=f"Event '{event}' not found in 2025 schedule"
            )

        event_name = event_info["EventName"]

        # Query MongoDB for prediction results
        query = {"Year": 2025, "GrandPrix": event_name, "IsPrediction": True}

        # Get lap data with predictions
        results = list(collection.find(query).sort("Position", 1))

        if not results:
            raise HTTPException(
                status_code=404, detail=f"No prediction results found for {event_name}"
            )

        # Get source years
        source_years = sorted(
            list(set(doc.get("SourceYear") for doc in results if "SourceYear" in doc))
        )

        # Process results into a format suitable for the frontend
        processed_results = []
        seen_drivers = set()

        for doc in results:
            driver = doc.get("Driver")

            # Skip drivers we've already processed (to avoid duplicates)
            if driver in seen_drivers:
                continue

            seen_drivers.add(driver)

            # Create a result object for this driver
            result = {
                "driver": driver,
                "position": doc.get("Position"),
                "team": doc.get("Team"),
                "lap_time": doc.get("LapTime"),
                "source_year": doc.get("SourceYear"),
                "source_event": doc.get("SourceEvent"),
            }

            processed_results.append(result)

        # Sort by position
        processed_results.sort(key=lambda x: x.get("position", 999))

        return {
            "event": event_name,
            "prediction_year": 2025,
            "source_years": source_years,
            "results": processed_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction results: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
