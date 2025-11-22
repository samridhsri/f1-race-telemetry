import streamlit as st
import mlflow
import mlflow.spark
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
import os
from datetime import datetime, timedelta
import requests

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def show_mlflow_dashboard():
    """Display MLflow experiment tracking and model management dashboard"""
    st.title("🔬 ML Experiment Tracking & Model Management")
    st.write("Monitor and compare F1 race prediction model experiments using MLflow")
    
    # Check MLflow connection
    if not check_mlflow_connection():
        st.error(f"Cannot connect to MLflow server at {MLFLOW_TRACKING_URI}")
        st.info("Make sure the MLflow server is running and accessible.")
        return
    
    # Create tabs for different views
    experiment_tab, model_registry_tab, artifacts_tab, metrics_tab = st.tabs([
        "🧪 Experiments", "📦 Model Registry", "📄 Artifacts", "📊 Metrics Comparison"
    ])
    
    with experiment_tab:
        show_experiments_view()
    
    with model_registry_tab:
        show_model_registry_view()
    
    with artifacts_tab:
        show_artifacts_view()
    
    with metrics_tab:
        show_metrics_comparison()

def check_mlflow_connection():
    """Check if MLflow server is accessible"""
    try:
        experiments = client.search_experiments()
        return True
    except Exception as e:
        st.error(f"MLflow connection error: {e}")
        return False

def show_experiments_view():
    """Display experiments and runs overview"""
    st.subheader("Experiment Overview")
    
    try:
        # Get all experiments
        experiments = client.search_experiments()
        
        if not experiments:
            st.info("No experiments found. Run a model training to create your first experiment!")
            return
        
        # Display experiments
        experiment_data = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            experiment_data.append({
                "Experiment Name": exp.name,
                "Experiment ID": exp.experiment_id,
                "Total Runs": len(runs),
                "Created": exp.creation_time,
                "Last Modified": exp.last_update_time if exp.last_update_time else exp.creation_time
            })
        
        exp_df = pd.DataFrame(experiment_data)
        if not exp_df.empty:
            # Convert timestamps
            exp_df["Created"] = pd.to_datetime(exp_df["Created"], unit='ms')
            exp_df["Last Modified"] = pd.to_datetime(exp_df["Last Modified"], unit='ms')
            
            st.dataframe(exp_df, use_container_width=True)
            
            # Select experiment to view runs
            selected_exp_name = st.selectbox(
                "Select Experiment to View Runs:",
                options=exp_df["Experiment Name"].tolist(),
                key="exp_selector"
            )
            
            if selected_exp_name:
                show_experiment_runs(selected_exp_name)
        
    except Exception as e:
        st.error(f"Error fetching experiments: {e}")

def show_experiment_runs(experiment_name):
    """Display runs for a specific experiment"""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        if not runs:
            st.info(f"No runs found for experiment: {experiment_name}")
            return
        
        st.subheader(f"Runs in {experiment_name}")
        
        # Prepare runs data
        runs_data = []
        for run in runs:
            run_data = {
                "Run ID": run.info.run_id[:8] + "...",  # Shortened for display
                "Run Name": run.data.tags.get("mlflow.runName", "Unnamed"),
                "Status": run.info.status,
                "Start Time": pd.to_datetime(run.info.start_time, unit='ms'),
                "Duration (min)": round((run.info.end_time - run.info.start_time) / (1000 * 60), 2) if run.info.end_time else None,
                "Race": run.data.tags.get("race_name", "Unknown"),
                "R² Score": run.data.metrics.get("r2_score", None),
                "RMSE": run.data.metrics.get("rmse", None),
                "Position Accuracy": run.data.metrics.get("position_accuracy_within_2", None),
                "Full Run ID": run.info.run_id
            }
            runs_data.append(run_data)
        
        runs_df = pd.DataFrame(runs_data)
        
        if not runs_df.empty:
            # Display runs table
            display_df = runs_df.drop("Full Run ID", axis=1)
            st.dataframe(display_df, use_container_width=True)
            
            # Run selection for detailed view
            selected_run_name = st.selectbox(
                "Select Run for Detailed View:",
                options=runs_df["Run Name"].tolist(),
                key="run_selector"
            )
            
            if selected_run_name:
                selected_run_id = runs_df[runs_df["Run Name"] == selected_run_name]["Full Run ID"].iloc[0]
                show_run_details(selected_run_id)
        
    except Exception as e:
        st.error(f"Error fetching runs: {e}")

def show_run_details(run_id):
    """Display detailed information for a specific run"""
    try:
        run = client.get_run(run_id)
        
        st.subheader(f"Run Details: {run.data.tags.get('mlflow.runName', 'Unnamed')}")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parameters:**")
            if run.data.params:
                params_df = pd.DataFrame(list(run.data.params.items()), columns=["Parameter", "Value"])
                st.dataframe(params_df, use_container_width=True, hide_index=True)
            else:
                st.info("No parameters logged")
        
        with col2:
            st.write("**Metrics:**")
            if run.data.metrics:
                metrics_df = pd.DataFrame(list(run.data.metrics.items()), columns=["Metric", "Value"])
                metrics_df["Value"] = metrics_df["Value"].round(4)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("No metrics logged")
        
        # Tags
        st.write("**Tags:**")
        if run.data.tags:
            tags_df = pd.DataFrame(list(run.data.tags.items()), columns=["Tag", "Value"])
            st.dataframe(tags_df, use_container_width=True, hide_index=True)
        
        # Artifacts
        st.write("**Artifacts:**")
        try:
            artifacts = client.list_artifacts(run_id)
            if artifacts:
                artifact_names = [artifact.path for artifact in artifacts]
                st.write("Available artifacts:", ", ".join(artifact_names))
                
                # Display images if available
                for artifact in artifacts:
                    if artifact.path.endswith('.png'):
                        st.write(f"**{artifact.path}:**")
                        try:
                            # Note: In a real deployment, you'd want to serve artifacts through MLflow's artifact server
                            st.info(f"Artifact: {artifact.path} (view in MLflow UI)")
                        except:
                            st.info(f"Could not display {artifact.path}")
            else:
                st.info("No artifacts logged")
        except Exception as e:
            st.warning(f"Could not fetch artifacts: {e}")
        
    except Exception as e:
        st.error(f"Error fetching run details: {e}")

def show_model_registry_view():
    """Display registered models and their versions"""
    st.subheader("Model Registry")
    
    try:
        # Get registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            st.info("No registered models found. Train a model to register it!")
            st.write("**💡 Tip:** When you run a race prediction model, it will automatically be registered here.")
            
            # Show button to manually register models from recent runs
            if st.button("🔄 Check for Unregistered Models"):
                try:
                    # Get recent runs that might have models
                    experiments = client.search_experiments()
                    recent_models = []
                    
                    for exp in experiments:
                        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
                        for run in runs:
                            artifacts = client.list_artifacts(run.info.run_id)
                            model_artifacts = [a for a in artifacts if 'model' in a.path.lower()]
                            if model_artifacts:
                                recent_models.append({
                                    'run_id': run.info.run_id,
                                    'run_name': run.data.tags.get('mlflow.runName', f'Run {run.info.run_id[:8]}'),
                                    'race_name': run.data.tags.get('race_name', 'Unknown'),
                                    'model_path': model_artifacts[0].path
                                })
                    
                    if recent_models:
                        st.write("**Found unregistered models:**")
                        for model_info in recent_models:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"🤖 {model_info['run_name']} - {model_info['race_name']}")
                            with col2:
                                if st.button("Register", key=f"reg_{model_info['run_id'][:8]}"):
                                    try:
                                        model_name = f"F1RacePredictor_{model_info['race_name'].replace(' ', '_')}"
                                        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
                                        mlflow.register_model(model_uri, model_name)
                                        st.success(f"Registered model: {model_name}")
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Registration failed: {e}")
                    else:
                        st.info("No unregistered models found in recent runs")
                        
                except Exception as e:
                    st.error(f"Error checking for models: {e}")
            
            return
        
        # Display model information
        for model in registered_models:
            with st.expander(f"📦 {model.name}"):
                st.write(f"**Description:** {model.description or 'No description'}")
                st.write(f"**Created:** {pd.to_datetime(model.creation_timestamp, unit='ms')}")
                st.write(f"**Last Updated:** {pd.to_datetime(model.last_updated_timestamp, unit='ms')}")
                
                # Get model versions
                versions = client.search_model_versions(f"name='{model.name}'")
                
                if versions:
                    version_data = []
                    for version in versions:
                        version_data.append({
                            "Version": version.version,
                            "Stage": version.current_stage,
                            "Status": version.status,
                            "Created": pd.to_datetime(version.creation_timestamp, unit='ms'),
                            "Run ID": version.run_id[:8] + "..." if version.run_id else "N/A",
                            "Source": version.source
                        })
                    
                    versions_df = pd.DataFrame(version_data)
                    st.dataframe(versions_df, use_container_width=True, hide_index=True)
                    
                    # Model stage transition buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"Promote to Staging", key=f"staging_{model.name}"):
                            try:
                                latest_version = max([int(v.version) for v in versions])
                                client.transition_model_version_stage(
                                    name=model.name,
                                    version=str(latest_version),
                                    stage="Staging"
                                )
                                st.success(f"Model {model.name} v{latest_version} promoted to Staging")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error promoting model: {e}")
                    
                    with col2:
                        if st.button(f"Promote to Production", key=f"prod_{model.name}"):
                            try:
                                latest_version = max([int(v.version) for v in versions])
                                client.transition_model_version_stage(
                                    name=model.name,
                                    version=str(latest_version),
                                    stage="Production"
                                )
                                st.success(f"Model {model.name} v{latest_version} promoted to Production")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error promoting model: {e}")
                    
                    with col3:
                        if st.button(f"Archive", key=f"archive_{model.name}"):
                            try:
                                latest_version = max([int(v.version) for v in versions])
                                client.transition_model_version_stage(
                                    name=model.name,
                                    version=str(latest_version),
                                    stage="Archived"
                                )
                                st.success(f"Model {model.name} v{latest_version} archived")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error archiving model: {e}")
                else:
                    st.info("No versions found for this model")
        
    except Exception as e:
        st.error(f"Error fetching registered models: {e}")

def show_artifacts_view():
    """Display artifacts browser"""
    st.subheader("Artifacts Browser")
    
    try:
        # Get all experiments
        experiments = client.search_experiments()
        
        if not experiments:
            st.info("No experiments found.")
            return
        
        # Select experiment
        exp_names = [exp.name for exp in experiments]
        selected_exp = st.selectbox("Select Experiment:", exp_names)
        
        if selected_exp:
            experiment = client.get_experiment_by_name(selected_exp)
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs:
                # Select run
                run_names = [f"{run.data.tags.get('mlflow.runName', f'Run {run.info.run_id[:8]}')} ({run.info.run_id[:8]})" for run in runs]
                selected_run_display = st.selectbox("Select Run:", run_names)
                
                if selected_run_display:
                    # Extract run ID from display name
                    run_id = selected_run_display.split('(')[-1].replace(')', '')
                    full_run_id = next(run.info.run_id for run in runs if run.info.run_id.startswith(run_id))
                    
                    # List artifacts for selected run
                    artifacts = client.list_artifacts(full_run_id)
                    
                    if artifacts:
                        st.write("**Available Artifacts:**")
                        
                        artifact_types = {
                            'images': [a for a in artifacts if a.path.endswith(('.png', '.jpg', '.jpeg'))],
                            'models': [a for a in artifacts if 'model' in a.path.lower()],
                            'data': [a for a in artifacts if a.path.endswith(('.csv', '.json', '.txt'))],
                            'other': [a for a in artifacts if not any([
                                a.path.endswith(('.png', '.jpg', '.jpeg')),
                                'model' in a.path.lower(),
                                a.path.endswith(('.csv', '.json', '.txt'))
                            ])]
                        }
                        
                        # Display artifacts by type
                        for artifact_type, artifact_list in artifact_types.items():
                            if artifact_list:
                                st.write(f"**{artifact_type.title()}:**")
                                for artifact in artifact_list:
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.write(f"📄 {artifact.path}")
                                    with col2:
                                        if artifact.file_size:
                                            size_mb = artifact.file_size / (1024 * 1024)
                                            st.write(f"{size_mb:.2f} MB")
                                    with col3:
                                        # For images, try to display them
                                        if artifact.path.endswith('.png'):
                                            if st.button(f"View", key=f"view_{artifact.path}"):
                                                try:
                                                    # Download and display artifact
                                                    artifact_path = client.download_artifacts(full_run_id, artifact.path)
                                                    st.image(artifact_path, caption=artifact.path)
                                                except Exception as e:
                                                    st.error(f"Could not display {artifact.path}: {e}")
                                        elif artifact.path.endswith('.txt'):
                                            if st.button(f"View", key=f"view_{artifact.path}"):
                                                try:
                                                    artifact_path = client.download_artifacts(full_run_id, artifact.path)
                                                    with open(artifact_path, 'r') as f:
                                                        content = f.read()
                                                    st.text_area(f"Content of {artifact.path}:", content, height=200)
                                                except Exception as e:
                                                    st.error(f"Could not display {artifact.path}: {e}")
                    else:
                        st.info("No artifacts found for this run")
            else:
                st.info("No runs found in this experiment")
    
    except Exception as e:
        st.error(f"Error browsing artifacts: {e}")
    
    # Add link to MLflow UI as fallback
    st.markdown("---")
    mlflow_ui_url = MLFLOW_TRACKING_URI.replace("mlflow:5000", "localhost:5001")
    st.markdown(f"**[Open MLflow UI]({mlflow_ui_url})** for full artifact management")

def show_metrics_comparison():
    """Display metrics comparison across runs"""
    st.subheader("Metrics Comparison")
    
    try:
        # Get all experiments
        experiments = client.search_experiments()
        
        if not experiments:
            st.info("No experiments found.")
            return
        
        # Collect all runs across experiments
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", f"Run {run.info.run_id[:8]}"),
                    "experiment": exp.name,
                    "race_name": run.data.tags.get("race_name", "Unknown"),
                    "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
                    **run.data.metrics
                }
                all_runs.append(run_info)
        
        if not all_runs:
            st.info("No runs with metrics found.")
            return
        
        runs_df = pd.DataFrame(all_runs)
        
        # Select metrics to compare
        numeric_columns = runs_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_columns:
            st.info("No numeric metrics found to compare.")
            return
        
        selected_metrics = st.multiselect(
            "Select metrics to compare:",
            options=numeric_columns,
            default=["r2_score", "rmse"] if all(m in numeric_columns for m in ["r2_score", "rmse"]) else numeric_columns[:2]
        )
        
        if selected_metrics:
            # Create comparison visualizations
            for metric in selected_metrics:
                st.write(f"**{metric.replace('_', ' ').title()} Comparison**")
                
                # Filter runs that have this metric
                metric_runs = runs_df.dropna(subset=[metric])
                
                if metric_runs.empty:
                    st.info(f"No runs found with {metric}")
                    continue
                
                # Create bar chart
                fig = px.bar(
                    metric_runs.sort_values(metric, ascending=False),
                    x="run_name",
                    y=metric,
                    color="race_name",
                    title=f"{metric.replace('_', ' ').title()} by Run",
                    hover_data=["experiment", "start_time"]
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top 5 runs for this metric
                top_runs = metric_runs.nlargest(5, metric)[["run_name", "race_name", metric]]
                st.write(f"Top 5 runs by {metric}:")
                st.dataframe(top_runs, use_container_width=True, hide_index=True)
        
        # Correlation matrix if multiple metrics selected
        if len(selected_metrics) > 1:
            st.write("**Metrics Correlation Matrix**")
            corr_data = runs_df[selected_metrics].corr()
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                title="Metrics Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error comparing metrics: {e}")

# Main function to run the dashboard
if __name__ == "__main__":
    show_mlflow_dashboard() 