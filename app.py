# app.py (FINAL WORKING VERSION)
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point
import geopandas as gpd
from typing import List, Dict, Any
import os
import requests
from functools import lru_cache

# --- Configuration ---
MODEL_PATH = 'uttarakhand_fire_model_ultimate.pkl'
# Saare 8 factors jo humne train kiye hain
FEATURES = ['X_frp', 'slope', 'temp', 'humidity', 'wind_speed', 'fuel_dryness', 'pop_density', 'dist_to_road']
SPREAD_RESOLUTION_KM = 0.5

# --- Model Loading (FIXED: Load on startup to prevent import error) ---
ML_MODEL = None
print("⚠️ Model loading deferred until app startup complete. Ensure .pkl file is correct.")

app = FastAPI(title="Uttarakhand Forest Fire Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
def load_model_on_startup():
    """App start hone par model ko load karta hai."""
    global ML_MODEL
    try:
        import time
        start_time = time.time()
        # Final Model Load
        ML_MODEL = joblib.load(MODEL_PATH)
        load_time = time.time() - start_time
        print(f"✅ ML Model Loaded Successfully on Startup in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"❌ ERROR: Could not load ML model from {MODEL_PATH}. Check file name and existence. Error: {e}")
        ML_MODEL = None

# --- 1. Simulation Logic (Cellular Automaton + ML) ---

def simulate_spread(initial_fires_gdf: gpd.GeoDataFrame, model, hours_to_predict: int):
    """
    Simulates fire spread based on ML prediction probability and wind/slope bias.
    Updated to handle long-term predictions with daily weather updates.
    """
    if model is None:
        return {"error": "ML model not loaded for simulation."}, []

    DEGREE_OFFSET = 0.5 / 111.0 # 500m offset
    current_fires_df = initial_fires_gdf.copy()
    current_fires_df['t_hour'] = 0
    unique_cells = set(zip(current_fires_df.geometry.x.round(4), current_fires_df.geometry.y.round(4)))
    results = {"0h": initial_fires_gdf[['geometry', 'X_frp']].to_json()}
    logs = []

    # Seasonal weather patterns for Uttarakhand (simplified)
    def get_seasonal_weather(hour):
        day = hour // 24
        month = (day // 30) % 12  # Approximate month

        # Base weather by season
        if month in [11, 0, 1]:  # Winter (Dec-Feb)
            temp_base = np.random.uniform(5, 15)
            humidity_base = np.random.uniform(40, 70)
        elif month in [2, 3, 4]:  # Spring (Mar-May)
            temp_base = np.random.uniform(15, 25)
            humidity_base = np.random.uniform(30, 60)
        elif month in [5, 6, 7, 8, 9]:  # Monsoon (Jun-Oct)
            temp_base = np.random.uniform(20, 30)
            humidity_base = np.random.uniform(60, 90)
        else:  # Autumn (Nov)
            temp_base = np.random.uniform(10, 20)
            humidity_base = np.random.uniform(50, 80)

        # Daily variation
        temp = temp_base + np.random.uniform(-5, 5)
        humidity = np.clip(humidity_base + np.random.uniform(-10, 10), 10, 95)
        wind_speed = np.random.uniform(1, 12)

        return temp, humidity, wind_speed

    # Simulation Logic with daily weather updates
    for t in range(1, hours_to_predict + 1):
        newly_ignited = []
        spreading_fires = current_fires_df[current_fires_df['t_hour'] == t-1]
        if spreading_fires.empty: break

        # Update weather every 24 hours
        if t % 24 == 1 or t == 1:
            current_temp, current_humidity, current_wind_speed = get_seasonal_weather(t)
            current_wind_dir = np.random.uniform(200, 320)
            wind_rad = np.radians(current_wind_dir)
            wind_x_comp = np.cos(wind_rad)
            wind_y_comp = np.sin(wind_rad)

        for _, row in spreading_fires.iterrows():
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dx, dy in offsets:
                new_lon = row.geometry.x + dx * DEGREE_OFFSET
                new_lat = row.geometry.y + dy * DEGREE_OFFSET
                cell_coords = (round(new_lon, 4), round(new_lat, 4))

                if cell_coords in unique_cells: continue

                # Update features with current weather
                new_features_data = {
                    'X_frp': np.clip(row['X_frp'] * np.random.uniform(0.9, 1.1), 10, 250),
                    'slope': row['slope'],
                    'temp': current_temp,
                    'humidity': current_humidity,
                    'wind_speed': current_wind_speed,
                    'fuel_dryness': row['fuel_dryness'],
                    'pop_density': row['pop_density'],
                    'dist_to_road': row['dist_to_road'] + 500,
                }
                new_features_df = pd.DataFrame([new_features_data])

                ignition_prob = model.predict_proba(new_features_df[FEATURES])[0, 1]

                uphill_bias = (dx * row['slope'] / 45 + dy * row['slope'] / 45) * 0.1
                wind_alignment = (dx * wind_x_comp + dy * wind_y_comp)
                wind_bias_factor = wind_alignment * 0.25
                final_prob = np.clip(ignition_prob + wind_bias_factor + uphill_bias, 0.0, 1.0)

                if final_prob > 0.45:
                    new_fire = {
                        'geometry': Point(new_lon, new_lat), 't_hour': t, 'prob_spread': final_prob, **new_features_data
                    }
                    newly_ignited.append(new_fire)
                    unique_cells.add(cell_coords)

        # Process results
        if newly_ignited:
            new_fires_gdf = gpd.GeoDataFrame(newly_ignited, geometry='geometry', crs="EPSG:4326")
            results[f"{t}h"] = new_fires_gdf[['prob_spread', 'X_frp', 'geometry']].to_json()
            current_fires_df = pd.concat([current_fires_df, new_fires_gdf.drop(columns=['prob_spread'])], ignore_index=True)
            log_entry = f"🔥 Hour {t}: {len(newly_ignited)} new cells ignited."
            logs.append(log_entry)
        else:
            break

    return results, logs

# --- 2. API Endpoints Classes and Definitions ---
class FirePoint(BaseModel):
    type: str = "Feature"
    properties: Dict[str, Any]
    geometry: Dict[str, Any]

class ActiveFiresInput(BaseModel):
    type: str = "FeatureCollection"
    features: List[FirePoint]
    hours_to_predict: int = 3
    use_real_weather: bool = False

# --- Weather API Integration ---
@lru_cache(maxsize=32)
def get_weather_data(lat: float, lon: float):
    """Fetch real-time weather data from OpenWeatherMap API."""
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with actual key
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

@app.get("/")
def read_root():
    return HTMLResponse(content=open("index.html", encoding='utf-8').read(), status_code=200)

@app.get("/api/v1/historical-fires")
def get_historical_fires():
    return {
        "status": "Mock historical data provided.",
        "fires": [
            {"date": "2023-05-15", "location": "Nainital Forest", "area_ha": 500, "details": "Past data is currently mocked. Use FSI records for real data."}
        ]
    }

@app.post("/api/v1/predict-spread")
async def predict_spread_endpoint(request: ActiveFiresInput):
    if ML_MODEL is None:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ML Model not loaded on server. Cannot run simulation.")

    try:
        initial_fires_gdf = gpd.GeoDataFrame.from_features(request.dict(by_alias=True)['features'], crs="EPSG:4326")
        
        if initial_fires_gdf.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active fire points provided in the GeoJSON.")
        
        # MOCKING LIVE DATA FOR THE 8 MODEL INPUTS (Same as in index.html)
        initial_fires_gdf['X_frp'] = initial_fires_gdf['frp']
        initial_fires_gdf['slope'] = initial_fires_gdf['slope']

        # Integrate real weather if requested
        if request.use_real_weather:
            # Use the first fire's location for weather (assuming single fire for simplicity)
            lat = initial_fires_gdf.geometry.y.iloc[0]
            lon = initial_fires_gdf.geometry.x.iloc[0]
            weather = get_weather_data(lat, lon)
            if weather:
                initial_fires_gdf['temp'] = weather['temp']
                initial_fires_gdf['humidity'] = weather['humidity']
                initial_fires_gdf['wind_speed'] = weather['wind_speed']
                logs.append(f"🌤️ Real weather data used: Temp {weather['temp']}°C, Humidity {weather['humidity']}%, Wind {weather['wind_speed']} m/s")
            else:
                logs.append("⚠️ Weather API failed, using mock data.")
                initial_fires_gdf['temp'] = 32
                initial_fires_gdf['humidity'] = 25
                initial_fires_gdf['wind_speed'] = 8
        else:
            initial_fires_gdf['temp'] = 32
            initial_fires_gdf['humidity'] = 25
            initial_fires_gdf['wind_speed'] = 8

        initial_fires_gdf['fuel_dryness'] = 0.2
        initial_fires_gdf['pop_density'] = 5
        initial_fires_gdf['dist_to_road'] = 500

        simulation_results, logs = simulate_spread(
            initial_fires_gdf[FEATURES + ['geometry']],
            ML_MODEL,
            request.hours_to_predict
        )

        parsed_timesteps = {
            k: json.loads(v)
            for k, v in simulation_results.items()
        }

        return {"timesteps": parsed_timesteps, "logs": logs, "status": "Simulation Completed. Map data is ready."}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Simulation Error: {str(e)}")

# --- FINAL FIX FOR UVICORN IMPORT ISSUE ---
if __name__ == "__main__":
    import uvicorn
    # Ab uvicorn ko 'app:app' ki jagah direct 'app' object mil raha hai
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)