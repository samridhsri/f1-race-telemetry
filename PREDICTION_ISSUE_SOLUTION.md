# Race Prediction Issue - Complete Solution Guide

## Current Situation

✅ **MLflow**: Predictions ARE running and completing successfully
❌ **UI**: Results not showing on Race Predictions page
❌ **Files**: Prediction results not being saved to filesystem

## Root Cause

The prediction model is **crashing or timing out** before it can:
1. Return the predictions DataFrame to the UI
2. Save results to `/app/predictions/results/`
3. Update `st.session_state.model_predictions`

Evidence from logs:
- Training starts successfully
- Spark creates DataFrame with 60 rows
- Model training begins
- Then... **silence** (no "Step 6: Saving" log, no completion)

## Likely Issues

### 1. **Spark Memory Crash**
```
ERROR DAGScheduler: Failed to update accumulator
java.net.SocketException: Broken pipe
```
This suggests Spark is running out of memory or crashing during training.

### 2. **Timeout**
The model training takes too long and Streamlit/Spark times out.

### 3. **Unhandled Exception**
An error occurs during prediction but isn't being caught properly.

## Immediate Solutions

### Solution 1: Check MLflow for Completed Runs

Since MLflow shows the metrics, the data IS there:

1. Go to MLflow: **http://localhost:5001**
2. Find your experiment: "F1 Race Prediction"
3. Click on the most recent run
4. Check the "Artifacts" section
5. Look for the model and any saved data

### Solution 2: Increase Spark Memory

Edit `docker-compose.yml`:

```yaml
streamlit:
  ...
  environment:
    - SPARK_DRIVER_MEMORY=2g
    - SPARK_EXECUTOR_MEMORY=2g
  mem_limit: 2g  # Increase from 1g to 2g
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Solution 3: Use Simpler Model (Quick Fix)

The Gradient Boosting Tree model is heavy. Edit `race_prediction_model.py:708`:

```python
# BEFORE (Heavy)
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=target_col,
    maxIter=50,      # ← Reduce this
    maxDepth=3,
    stepSize=0.1,
)

# AFTER (Lighter)
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=target_col,
    maxIter=20,      # ← Reduced from 50
    maxDepth=2,       # ← Reduced from 3
    stepSize=0.1,
)
```

### Solution 4: Simplify Features

The model is using too many features. Edit `race_prediction_model.py:649`:

```python
# Use fewer numeric features for faster training
numeric_features = [
    'GridPosition',
    'AvgLapTime',
    'BestLapTime',
    'AvgSpeed'
]
# Remove: LapTimeConsistency, MaxSpeed, PitStopCount, weather features, etc.
```

## Long-term Fix: Add Robust Error Handling

The prediction needs better error handling and intermediate saves. Here's what should be added:

### 1. Save After Each Step

```python
# After data fetching
save_intermediate_data(historical_data, "01_raw_data.pkl")

# After preprocessing
save_intermediate_data(processed_data, "02_processed_data.pkl")

# After model training
save_model(model, "03_trained_model")

# After predictions
save_prediction_results(predictions, race_name)
```

### 2. Add Timeout Protection

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Prediction timed out")

# Set 5 minute timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minutes

try:
    predictions = run_prediction_model(...)
finally:
    signal.alarm(0)  # Cancel alarm
```

### 3. Return Partial Results

Even if full prediction fails, return what we have:

```python
try:
    # Full prediction
    predictions = predict_race_outcome(...)
    return predictions
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    # Return at least the 2026 driver dataset
    return create_2026_driver_dataset(processed_data)
```

## Debugging Steps

### 1. Check Container Resources

```bash
docker stats streamlit
```

Look for:
- Memory usage near limit
- CPU at 100%

### 2. Watch Logs in Real-Time

```bash
docker logs -f streamlit
```

Run a prediction and watch for:
- Where it stops
- Any error messages
- Memory warnings

### 3. Check Spark UI (if enabled)

If Spark UI is running, check:
- Job progress
- Failed stages
- Memory usage

### 4. Test with Minimal Data

Try predicting with just 1 year:
```python
source_years_str = "2024"  # Instead of "2023,2024,2025"
```

## Quick Workaround: Manual Result Retrieval

Since MLflow has the data, you can manually create a prediction result:

1. **From MLflow Artifacts:**
   - Download the model artifacts
   - Load the prediction data
   - Create a CSV manually

2. **Place in UI:**
   ```python
   # In Streamlit, manually load:
   import pandas as pd
   predictions = pd.read_csv('/path/to/predictions.csv')
   st.session_state.model_predictions = predictions
   ```

## Recommended Action Plan

1. **Immediate** (5 min):
   - Increase Streamlit container memory to 2GB
   - Restart containers

2. **Short-term** (15 min):
   - Reduce model complexity (maxIter=20, maxDepth=2)
   - Test with single year data

3. **Medium-term** (1 hour):
   - Add comprehensive error handling
   - Add intermediate checkpoints
   - Save predictions after each step

4. **Long-term** (Later):
   - Move heavy ML training to separate service
   - Use async job queue (Celery)
   - Cache trained models
   - Pre-compute predictions

## Testing the Fix

After applying changes:

```bash
# 1. Restart
docker-compose restart streamlit

# 2. Watch logs
docker logs -f streamlit

# 3. In another terminal, check resources
docker stats

# 4. Run prediction (use simple race like Bahrain)

# 5. Monitor completion
```

Expected timeline with fixes:
- Light model: 1-2 minutes
- Original model: 3-5 minutes
- Should complete without crashing

## Success Indicators

✅ See in logs: "Step 6: Saving prediction results..."
✅ See in logs: "Prediction process completed successfully"
✅ Files appear in: `/app/predictions/results/`
✅ Results show on UI automatically
✅ MLflow shows completed run with artifacts

## Still Not Working?

If predictions still fail after all fixes:

1. **Simplest possible test:**
   ```python
   # Comment out entire ML pipeline
   # Return dummy data
   def run_prediction_model(race_name, source_years_str):
       dummy_data = pd.DataFrame({
           'FullName': ['Max Verstappen', 'Lewis Hamilton'],
           'TeamName': ['Red Bull', 'Ferrari'],
           'PredictedPosition': [1, 2],
           'GrandPrix': [race_name, race_name],
           'Year': [2026, 2026]
       })
       return dummy_data
   ```

2. **If dummy works:** Issue is in ML pipeline
3. **If dummy fails:** Issue is in Streamlit state management

---

**Next Steps:** Try Solution 2 (increase memory) + Solution 3 (reduce model complexity) first.
