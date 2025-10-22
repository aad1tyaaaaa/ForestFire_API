[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

# 🔥 Uttarakhand Forest Fire Prediction API

> A cutting-edge FastAPI-based web application for predicting and simulating forest fire spread in Uttarakhand, India. Leveraging machine learning and cellular automaton models to forecast fire propagation based on environmental factors. 🌍📈

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🧠 Model Details](#-model-details)
- [🔗 Resources](#-resources)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [🐛 Issues and TODO](#-issues-and-todo)
- [📞 Contact](#-contact)

## ✨ Features

- **🔥 Real-time Fire Spread Simulation**: Advanced cellular automaton combined with ML-based probability models for accurate predictions.
- **🖥️ Interactive Web UI**: Intuitive interface for data input and visualization of fire spread predictions.
- **🔗 RESTful API**: Robust endpoints for historical data retrieval and fire spread simulations.
- **🌤️ Weather Integration**: Seamless integration with OpenWeatherMap API for real-time weather data.
- **🗺️ GeoJSON Support**: Full support for GeoJSON input/output, enabling easy integration with mapping tools like Leaflet or Mapbox.
- **📊 Comprehensive Logging**: Detailed simulation logs for monitoring and analysis.
- **⚡ High Performance**: Optimized for quick simulations with scalable architecture.

## 📸 Screenshots

### Web Interface
![Web Interface Screenshot](screenshots/web_interface.png)
*Main dashboard for inputting fire data and viewing predictions.*

### API Documentation
![API Docs Screenshot](screenshots/api_docs.png)
*Interactive Swagger UI for exploring API endpoints.*

### Simulation Results
![Simulation Results Screenshot](screenshots/simulation_results.png)
*Visualization of fire spread prediction over time.*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher 🐍
- pip (Python package manager) 📦
- Git (for cloning the repository) 🗂️

### Steps

1. **📥 Clone the repository**:
   ```bash
   git clone https://github.com/aad1tyaaaaa/ForestFire_API.git
   cd ForestFire_API
   ```

2. **📦 Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **🤖 Set up the ML model**:
   - Ensure `uttarakhand_fire_model_ultimate.pkl` is in the root directory.
   - If not present, train the model using the hardcoded dataset:
     ```bash
     python generate_model.py
     ```
     *(Note: Uses hardcoded dataset.csv with 46 samples representing Uttarakhand forest fire conditions)*

4. **🔑 Configure API keys** (optional, for real weather data):
   - Obtain an API key from [OpenWeatherMap](https://openweathermap.org/api).
   - Update `app.py`:
     ```python
     API_KEY = "your_actual_api_key_here"
     ```

## 🚀 Usage

### Running the Application

1. **▶️ Start the server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **🌐 Access the application**:
   - **Web Interface**: Visit `http://127.0.0.1:8000` for the interactive UI.
   - **API Documentation**: Explore `http://127.0.0.1:8000/docs` for Swagger UI docs.

### API Endpoints

#### GET /
- **📄 Description**: Serves the main HTML interface.
- **📤 Response**: HTML content for the web application.

#### GET /api/v1/historical-fires
- **📊 Description**: Fetches mock historical fire data for Uttarakhand.
- **📤 Response**: JSON array of historical fire records.

#### POST /api/v1/predict-spread
- **🔮 Description**: Executes fire spread simulation based on provided data.
- **📥 Request Body** (GeoJSON):
  ```json
  {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "frp": 120,
          "slope": 35
        },
        "geometry": {
          "type": "Point",
          "coordinates": [78.10, 30.05]
        }
      }
    ],
    "hours_to_predict": 4,
    "use_real_weather": false
  }
  ```
- **📤 Response**: JSON with timesteps, simulation logs, and status.

### 🧪 Testing

- **Sample Data**: Use `test_request.json` for testing.
- **Curl Example**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/predict-spread \
       -H "Content-Type: application/json" \
       -d @test_request.json
  ```

## 📁 Project Structure

```
ForestFire_API/
├── app.py                          # 🚀 Main FastAPI application
├── generate_model.py               # 🤖 ML model training script
├── index.html                      # 🖥️ Frontend UI
├── requirements.txt                # 📦 Python dependencies
├── uttarakhand_fire_model_ultimate.pkl  # 💾 Trained ML model
├── test_request.json               # 📄 Sample API payload
├── TODO.md                         # 📝 Development notes
└── README.md                       # 📖 This file
```

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier 🌳
- **Input Features**: All 8 environmental factors (X_frp, slope, temp, humidity, wind_speed, fuel_dryness, pop_density, dist_to_road)
- **Simulation Engine**: Cellular Automaton with ML-driven probabilities and long-term predictions
- **Environmental Factors**: Real-time weather integration, seasonal patterns, wind direction, slope gradients
- **Accuracy**: AUC 1.0000 on test data with 46 Uttarakhand-specific samples

## 🔗 Resources

### 📚 Documentation & Libraries
- [FastAPI Official Docs](https://fastapi.tiangolo.com/) - Web framework documentation
- [GeoPandas](https://geopandas.org/) - Geospatial data manipulation
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework
- [Shapely](https://shapely.readthedocs.io/) - Geometric operations
- [OpenWeatherMap API](https://openweathermap.org/api) - Weather data integration

### 🗺️ Related Datasets & Research
- [Forest Survey of India (FSI)](https://fsi.nic.in/) - Official forest fire statistics
- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) - Global fire data
- [Uttarakhand Forest Department](https://uttarakhandforest.org/) - Regional fire management
- [Research Paper: Forest Fire Prediction Models](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/forest-fire-prediction) - Academic resources

### 🛠️ Development Tools
- [Python](https://www.python.org/) - Programming language
- [Jupyter Notebook](https://jupyter.org/) - Data analysis and prototyping
- [VS Code](https://code.visualstudio.com/) - Recommended IDE
- [Postman](https://www.postman.com/) - API testing tool

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/amazing-feature`
3. 💾 Commit changes: `git commit -m 'Add amazing feature'`
4. 🚀 Push to branch: `git push origin feature/amazing-feature`
5. 📤 Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure compatibility with Python 3.8+

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. ⚖️

## 🙏 Acknowledgments

- **Data Sources**: Uttarakhand forest fire records and environmental datasets 📊
- **Libraries**: FastAPI, GeoPandas, Random Forest (scikit-learn), and other open-source tools 🛠️
- **Inspiration**: Real-world fire prediction research and environmental monitoring initiatives 🌟
- **Community**: Contributors and users who help improve this project 🤝

## 🐛 Issues and TODO

- See `TODO.md` for current issues and planned enhancements, including long-term predictions and UI improvements. 📋
- Report bugs or request features via [GitHub Issues](https://github.com/aad1tyaaaaa/ForestFire_API/issues).

## 📞 Contact

- **Authors**:
  - Aaditya Jaiswar ([GitHub](https://github.com/aad1tyaaaaa)) - Email: aadityaaaaa.jaiswar@gmail.com
  - Prathamesh Aarya ([GitHub](https://github.com/prathameshAarya12)) - Email: prathamesh.18216@sakec.ac.in


---

⭐ Star this repo if you find it useful! Contributions and feedback are always welcome. 🌟
