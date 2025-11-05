# Uttarakhand Forest Fire Prediction API Documentation

## Project Overview

This API provides advanced forest fire prediction and simulation capabilities specifically tailored for Uttarakhand, India. The system combines machine learning models with cellular automaton algorithms to forecast fire spread based on multiple environmental factors.

### Key Features
- **Machine Learning Prediction**: Random Forest classifier trained on 8 environmental factors
- **Cellular Automaton Simulation**: Spatiotemporal fire spread modeling with long-term predictions
- **Real-time Weather Integration**: OpenWeatherMap API integration with seasonal weather patterns
- **Interactive Web Interface**: Leaflet-based map visualization with charts and analytics
- **RESTful API**: Comprehensive endpoints for predictions and historical data
- **GeoJSON Support**: Full geospatial data input/output compatibility

### Technology Stack
- **Backend**: FastAPI (Python)
- **ML Framework**: Random Forest, scikit-learn
- **Geospatial Processing**: GeoPandas, Shapely
- **Data Processing**: Pandas, NumPy
- **Weather API**: OpenWeatherMap
- **Frontend**: HTML5, JavaScript, Leaflet, Chart.js
- **Styling**: Tailwind CSS, Custom CSS

### Project Structure
```
ForestFire_API/
├── app.py                          # Main FastAPI application
├── generate_model.py               # ML model training script
├── index.html                      # Web interface
├── requirements.txt                # Python dependencies
├── api_documentation.md            # This documentation
├── uttarakhand_fire_model_ultimate.pkl  # Trained ML model
├── test_request.json               # Sample API request
├── README.md                       # Project README
└── screenshots/                    # UI screenshots
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ForestFire_API
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ML model file exists:
   ```bash
   python generate_model.py
   ```

4. Configure weather API (optional):
   - Obtain API key from [OpenWeatherMap](https://openweathermap.org/api)
   - Update `API_KEY` variable in `app.py`

5. Start the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Accessing the Application
- **Web Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs (Swagger UI)
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc

## API Endpoints

### 1. GET /

**Description**: Serves the main web interface for interactive fire prediction and visualization.

**Endpoint**: `GET /`

**Response**: HTML content (index.html)

**Example**:
```bash
curl http://127.0.0.1:8000/
```

**Response**: Returns the complete HTML page with Leaflet map interface.

---

### 2. GET /api/v1/historical-fires

**Description**: Retrieves mock historical fire data for Uttarakhand region. Currently returns static sample data for demonstration purposes.

**Endpoint**: `GET /api/v1/historical-fires`

**Response Format**: JSON

**Response Schema**:
```json
{
  "status": "string",
  "fires": [
    {
      "date": "string",
      "location": "string",
      "area_ha": "number",
      "details": "string"
    }
  ]
}
```

**Example Request**:
```bash
curl http://127.0.0.1:8000/api/v1/historical-fires
```

**Example Response**:
```json
{
  "status": "Mock historical data provided.",
  "fires": [
    {
      "date": "2023-05-15",
      "location": "Nainital Forest",
      "area_ha": 500,
      "details": "Past data is currently mocked. Use FSI records for real data."
    }
  ]
}
```

**Status Codes**:
- `200`: Success
- `500`: Internal server error

---

### 3. POST /api/v1/predict-spread

**Description**: Executes fire spread simulation based on provided active fire data and environmental parameters. Uses ML model for ignition probability calculation and cellular automaton for spatial spread modeling.

**Endpoint**: `POST /api/v1/predict-spread`

**Request Format**: JSON (GeoJSON FeatureCollection)

**Request Schema**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "frp": "number",
        "slope": "number"
      },
      "geometry": {
        "type": "Point",
        "coordinates": ["number", "number"]
      }
    }
  ],
  "hours_to_predict": "integer",
  "use_real_weather": "boolean"  // Optional, defaults to false
}
```

**Request Parameters**:
- `features`: Array of GeoJSON features representing active fire points
- `frp`: Fire Radiative Power (intensity measure)
- `slope`: Terrain slope in degrees
- `coordinates`: [longitude, latitude] of fire location
- `hours_to_predict`: Number of hours to simulate (1-100+)
- `use_real_weather`: Whether to fetch real weather data (requires API key)

**Response Format**: JSON

**Response Schema**:
```json
{
  "timesteps": {
    "0h": "GeoJSON string",
    "1h": "GeoJSON string",
    ...
  },
  "logs": ["string"],
  "status": "string"
}
```

**Response Fields**:
- `timesteps`: Object containing GeoJSON strings for each time step
- `logs`: Array of simulation log messages
- `status`: Overall simulation status message

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-spread \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Example Response**:
```json
{
  "timesteps": {
    "0h": "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"properties\":{\"X_frp\":120,\"prob_spread\":1.0},\"geometry\":{\"type\":\"Point\",\"coordinates\":[78.1,30.05]}}]}",
    "1h": "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"properties\":{\"X_frp\":120,\"prob_spread\":0.75},\"geometry\":{\"type\":\"Point\",\"coordinates\":[78.1,30.05]}}]}"
  },
  "logs": [
    "🔥 Hour 1: 3 new cells ignited."
  ],
  "status": "Simulation Completed. Map data is ready."
}
```

**Error Responses**:
- `400`: Bad Request - No active fire points provided or invalid GeoJSON
- `503`: Service Unavailable - ML model not loaded
- `500`: Internal Server Error - Simulation failure

## Machine Learning Model Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Input Features**: 8 environmental factors
- **Training Data**: Hardcoded CSV dataset (46 samples) representing Uttarakhand forest fire conditions
- **Performance**: AUC 1.0000 on test data

### Input Features
1. **X_frp**: Fire Radiative Power (fire intensity)
2. **slope**: Terrain slope in degrees
3. **temp**: Temperature in Celsius
4. **humidity**: Relative humidity percentage
5. **wind_speed**: Wind speed in m/s
6. **fuel_dryness**: Fuel moisture content (0-1 scale)
7. **pop_density**: Population density
8. **dist_to_road**: Distance to nearest road in meters

### Prediction Logic
- **Base Probability**: 0.02 (2% base ignition chance)
- **Factor Multipliers**:
  - Slope >25°: +15%
  - High temp (>30°C) + Low humidity (<30%): +20%
  - Very dry fuel (<0.2 moisture): +15%
  - High wind (>10 m/s): +10%
  - Near road (<500m) or high population: +18%

## Simulation Engine

### Cellular Automaton Algorithm
- **Resolution**: 0.5 km grid cells
- **Neighborhood**: 8-directional (Moore neighborhood)
- **Time Step**: 1 hour
- **Spread Factors**:
  - Wind direction and speed
  - Terrain slope and aspect
  - Fuel dryness
  - Distance decay

### Weather Integration
- **API**: OpenWeatherMap
- **Parameters**: Temperature, humidity, wind speed
- **Fallback**: Seasonal weather patterns (Winter: Dec-Feb, Spring: Mar-May, Monsoon: Jun-Oct, Autumn: Nov)
- **Update Frequency**: Every 24 hours during long-term simulations
- **Caching**: LRU cache for 32 locations to reduce API calls

## Usage Examples

### Basic Prediction
```python
import requests

url = "http://127.0.0.1:8000/api/v1/predict-spread"
payload = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"frp": 120, "slope": 35},
        "geometry": {"type": "Point", "coordinates": [78.10, 30.05]}
    }],
    "hours_to_predict": 4
}

response = requests.post(url, json=payload)
result = response.json()
```

### With Real Weather
```python
payload["use_real_weather"] = True
# Requires OpenWeatherMap API key configured in app.py
```

## Error Handling

### Common Error Scenarios
1. **Model Not Loaded**: Ensure `uttarakhand_fire_model_ultimate.pkl` exists
2. **Invalid GeoJSON**: Validate input format matches schema
3. **Weather API Failure**: Falls back to mock weather data
4. **Memory Issues**: Large simulations may require more RAM

### Logging
- All simulations generate detailed logs
- Logs include cell ignition counts and timestamps
- Accessible via API response and web interface

## Performance Considerations

### Optimization Features
- **Caching**: Weather data cached for 32 locations
- **Vectorized Operations**: NumPy/Pandas for efficient computation
- **Resolution Limits**: 0.5km cells balance accuracy vs performance
- **Early Termination**: Stops if no new cells ignite

### System Requirements
- **RAM**: 4GB+ recommended for large simulations
- **CPU**: Multi-core processor for parallel processing
- **Storage**: ~100MB for model and dependencies

## Security Considerations

### API Security
- CORS enabled for web interface
- Input validation for all endpoints
- No authentication required (development/demo purposes)

### Data Privacy
- No user data stored
- Weather API calls may be logged by third-party service
- All processing done locally

## Future Enhancements

### Planned Features
- Real historical data integration
- Multi-fire scenario support
- Advanced visualization options
- Mobile application
- Real-time satellite data integration

### API Extensions
- Batch prediction endpoints
- Custom model training
- Historical analysis tools
- Risk assessment reports

## Support & Contributing

### Getting Help
- Check the README.md for setup instructions
- Review Swagger UI at `/docs` for interactive testing
- Examine `test_request.json` for sample payloads

### Development
- Follow PEP 8 coding standards
- Add tests for new features
- Update documentation for API changes
- Use GitHub Issues for bug reports

---

**Version**: 1.0.0
**Last Updated**: 2024
**License**: MIT
**Contact**: [Project Repository](https://github.com/aad1tyaaaaa/ForestFire_API)
