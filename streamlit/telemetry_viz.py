import os
import streamlit as st
import pandas as pd
import pymongo
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# MongoDB connection settings
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/")
DB_NAME = os.environ.get("DB_NAME", "f1db")


# Custom styling for charts and controls
def load_viz_styles():
    st.markdown(
        """
    <style>
        .chart-card {
            background-color: #1a1a1a;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #333;
        }
        .chart-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: white;
            text-align: center;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# MongoDB connection function
@st.cache_resource
def get_mongodb_client():
    return pymongo.MongoClient(MONGO_URI)


# Function to fetch telemetry data for a specific session
def fetch_session_telemetry(year: int, event: str, session_type: str) -> pd.DataFrame:
    try:
        client = get_mongodb_client()
        db = client[DB_NAME]
        telemetry_collection = db["telemetry"]

        # Query MongoDB for telemetry data (lap-level data)
        query = {"Year": year, "GrandPrix": event, "SessionType": session_type}

        # Select all available fields - don't use projection to see what's actually there
        cursor = telemetry_collection.find(query)
        data = list(cursor)

        # Convert to DataFrame
        if data:
            df = pd.DataFrame(data)
            # Remove _id column
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)
            
            # Check if Position column exists and has data
            if "Position" not in df.columns or (df["Position"].isna().all() if "Position" in df.columns else True):
                # For Race sessions, try to get Position from race_results collection
                if session_type == "Race":
                    try:
                        race_results_collection = db["race_results"]
                        results_query = {"Year": year, "GrandPrix": event, "SessionType": session_type}
                        results_data = list(race_results_collection.find(results_query, {"Driver": 1, "Position": 1, "DriverNumber": 1}))
                        
                        if results_data:
                            # Create a mapping of Driver to Position from race results
                            results_df = pd.DataFrame(results_data)
                            if "_id" in results_df.columns:
                                results_df = results_df.drop("_id", axis=1)
                            
                            # Try to merge Position from race results
                            if "Driver" in df.columns and "Driver" in results_df.columns:
                                # Merge on Driver to get Position
                                df = df.merge(
                                    results_df[["Driver", "Position"]].drop_duplicates(subset=["Driver"]),
                                    on="Driver",
                                    how="left",
                                    suffixes=("", "_from_results")
                                )
                                # Use Position from results if original Position is missing
                                if "Position_from_results" in df.columns:
                                    df["Position"] = df["Position"].fillna(df["Position_from_results"])
                                    df = df.drop("Position_from_results", axis=1)
                    except Exception as e:
                        # If merging fails, just continue with empty Position
                        pass
                
                # If Position is still not available, add empty column
                if "Position" not in df.columns:
                    df["Position"] = None
                    if session_type != "Race":
                        st.warning("⚠️ Position field not found in telemetry data. This is normal for Practice and Qualifying sessions.")
                        st.info("💡 Position tracking is typically only available during Race sessions.")
                elif df["Position"].isna().all():
                    st.warning("⚠️ Position data is not available for this session.")
                    if session_type != "Race":
                        st.info("💡 Position tracking is typically only available during Race sessions.")
            
            # Select only the fields we need for visualization
            desired_fields = [
                "LapNumber", "Position", "Driver", "Team", "TeamID", 
                "LapTime", "Compound", "Stint", "TyreLife", "Time", 
                "DriverNumber", "AverageSpeed"
            ]
            
            # Only include columns that exist in the dataframe
            available_fields = [f for f in desired_fields if f in df.columns]
            df = df[available_fields]
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching telemetry data: {e}")
        return pd.DataFrame()


# Function to create team colors mapping
def create_team_colors() -> Dict[str, str]:
    return {
        "Red Bull Racing": "#3671C6",
        "Ferrari": "#F91536",
        "Aston Martin": "#5E8FAA",
        "McLaren": "#F58020",
        "Mercedes": "#6CD3BF",
        "RB": "#C8C8C8",
        "Haas F1 Team": "#B6BABD",
        "Williams": "#37BEDD",
        "Alpine": "#2293D1",
        "Kick Sauber": "#00CF46",
        # Default for any unknown team
        "Unknown": "#FFFFFF",
    }


# Function to create the scatter plot
def create_position_lap_scatter(
    df: pd.DataFrame, team_colors: Dict[str, str]
) -> go.Figure:
    if df.empty:
        return go.Figure()

    # CRITICAL FIX: Check if Position column exists and has data
    if "Position" not in df.columns:
        st.error("❌ Position data is not available in the telemetry data.")
        st.info("💡 **Tip:** Position data is only available for Race sessions, not Practice or Qualifying.")
        st.info("📊 For Practice/Qualifying sessions, try the 'Lap Comparison' or 'Track Speed Heatmap' visualizations instead.")
        return go.Figure()
    
    # Filter out rows where Position is null or NaN
    df_with_position = df[df["Position"].notna()].copy()
    
    if df_with_position.empty:
        st.warning("⚠️ No position data found for this session.")
        st.info("💡 **Tip:** Position tracking is typically only available during Race sessions.")
        st.info("📊 For this session type, try other visualizations like 'Lap Comparison' or 'Track Speed Heatmap'.")
        return go.Figure()

    # Add a custom hover template with more information
    hover_template = (
        "<b>%{customdata[0]}</b> (#%{customdata[1]})<br>"
        + "Team: %{customdata[2]}<br>"
        + "Position: %{y}<br>"
        + "Lap: %{x}<br>"
        + "Lap Time: %{customdata[3]}<br>"
        + "Tyre: %{customdata[4]} (Stint: %{customdata[5]}, Life: %{customdata[6]})<br>"
        + "Avg Speed: %{customdata[7]} km/h"
    )

    # Prepare figure
    fig = go.Figure()

    # Get unique drivers from filtered data
    drivers = df_with_position["Driver"].unique()

    # Add a trace for each driver
    for driver in drivers:
        driver_df = df_with_position[df_with_position["Driver"] == driver]

        if not driver_df.empty:
            # Get the team name (or Unknown if not available)
            team = (
                driver_df["Team"].iloc[0] if "Team" in driver_df.columns else "Unknown"
            )

            # Get color based on team name
            color = team_colors.get(team, team_colors["Unknown"])

            # Create custom data for hover info - with safe field access
            custom_data_cols = []
            for col in ["Driver", "DriverNumber", "Team", "LapTime", "Compound", "Stint", "TyreLife", "AverageSpeed"]:
                if col in driver_df.columns:
                    custom_data_cols.append(col)
                else:
                    # Add a placeholder column if the field doesn't exist
                    driver_df[col] = "N/A"
                    custom_data_cols.append(col)
            
            custom_data = driver_df[custom_data_cols].values

            # Add scatter trace for this driver - markers only, no lines
            fig.add_trace(
                go.Scatter(
                    x=driver_df["LapNumber"],
                    y=driver_df["Position"],
                    mode="markers",
                    name=driver,
                    marker=dict(color=color, size=10),
                    customdata=custom_data,
                    hovertemplate=hover_template,
                )
            )

            text_positions = driver_df["Position"] + 0.4

            # Add text labels for each point with adjusted position
            fig.add_trace(
                go.Scatter(
                    x=driver_df["LapNumber"],
                    y=text_positions,
                    mode="text",
                    text=driver,
                    textposition="middle center",
                    textfont=dict(family="Arial", size=9, color=color),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Update layout
    fig.update_layout(
        title="Driver Positions by Lap",
        xaxis_title="Lap Number",
        yaxis_title="Position",
        # Reverse y-axis so position 1 is at the top
        yaxis=dict(autorange="reversed", tickmode="linear", tick0=1, dtick=1),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor="#2c2c2c",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
    )

    return fig


# Main visualization function to be called from app.py
def show_telemetry_visualization(year: int, event: str, session_type: str):
    # Initialize telemetry session state variables
    if "telemetry_data" not in st.session_state:
        st.session_state.telemetry_data = None

    # Load custom styles
    load_viz_styles()

    st.markdown("---")
    st.subheader("Telemetry Visualization")

    # Button to load telemetry data for visualization
    load_viz_data = st.button(
        "Load Telemetry Data for Visualization", use_container_width=True
    )

    # If button is clicked or telemetry data is already loaded, fetch and display
    if load_viz_data or st.session_state.telemetry_data is not None:
        if load_viz_data:
            with st.spinner(f"Loading telemetry data for visualization..."):
                # Fetch the telemetry data
                telemetry_data = fetch_session_telemetry(year, event, session_type)

                if not telemetry_data.empty:
                    st.session_state.telemetry_data = telemetry_data
                    st.success(f"Loaded {len(telemetry_data)} telemetry records")
                else:
                    st.error("No telemetry data found for this session")

        # If we have telemetry data, show the visualization
        if (
            st.session_state.telemetry_data is not None
            and not st.session_state.telemetry_data.empty
        ):
            # Chart section
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="chart-title">Position by Lap</div>', unsafe_allow_html=True
            )

            # Create team colors mapping
            team_colors = create_team_colors()

            # Create plot area
            plot_container = st.container()

            # Show the scatter plot
            with plot_container:
                fig = create_position_lap_scatter(
                    st.session_state.telemetry_data, team_colors
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Optionally show data table
            with st.expander("View Raw Telemetry Data"):
                sort_columns = ["LapNumber"]
                if "Position" in st.session_state.telemetry_data.columns:
                    if st.session_state.telemetry_data["Position"].notna().any():
                        sort_columns.append("Position")
                if "Driver" in st.session_state.telemetry_data.columns and "Position" not in sort_columns:
                    sort_columns.append("Driver")

                st.dataframe(
                    st.session_state.telemetry_data.sort_values(by=sort_columns),
                    use_container_width=True,
                )
