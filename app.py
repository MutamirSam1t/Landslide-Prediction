from flask import Flask, render_template, request
import pickle
import pandas as pd
import ee

# Initialize Flask app
app = Flask(__name__)

# Initialize Earth Engine
ee.Authenticate(force=True)
ee.Initialize(project='ee-moinulhossain00')

# Load your model
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# List of 8 features for prediction
feature_names = [
    "ELEVATION", "SLOPE", "ASPECT", "TWI", 
    "SPI", "NDVI", "RAINFALL", "LANDUSE"
]

def get_earth_engine_features(lat, lon):
    """
    Fetches the required features from Google Earth Engine for a given latitude and longitude.
    """
    point = ee.Geometry.Point([lon, lat])

    # Get the DEM (Digital Elevation Model) for the location
    dem = ee.Image("USGS/SRTMGL1_003")

    # Calculate the slope from the DEM
    slope = ee.Terrain.slope(dem)

    # Calculate the aspect (direction of the slope)
    aspect = ee.Terrain.aspect(dem)

    # Calculate flow accumulation
    try:
        flow_accumulation = ee.Algorithms.Terrain.flowAccumulation(dem)
    except AttributeError:
        flow_accumulation = ee.Image(0)
        print("Flow accumulation could not be computed.")

    # Calculate Topographic Wetness Index (TWI)
    try:
        twi = flow_accumulation.divide(slope.tan()).log()
    except AttributeError:
        twi = ee.Image(0)
        print("TWI could not be computed.")

    # Calculate Stream Power Index (SPI)
    try:
        spi = flow_accumulation.multiply(slope)
    except AttributeError:
        spi = ee.Image(0)
        print("SPI could not be computed.")

    # Get NDVI (Normalized Difference Vegetation Index) using MODIS data
    try:
        ndvi = ee.ImageCollection("MODIS/006/MOD13A1").filterDate('2021-01-01', '2021-12-31').mean().select('NDVI')
    except AttributeError:
        ndvi = ee.Image(0)
        print("NDVI could not be computed.")

    # Get Rainfall data (using CHIRPS dataset for global rainfall)
    try:
        rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate('2021-01-01', '2021-12-31').mean()
    except AttributeError:
        rainfall = ee.Image(0)
        print("Rainfall data could not be computed.")

    # Land Use data (use a global land use map like ESA or GlobCover)
    try:
        landuse = ee.Image("ESA/WorldCover/v100/2020").clip(point.buffer(1000))  # Clip to a 1km buffer around the point
    except AttributeError:
        landuse = ee.Image(0)
        print("Land use data could not be retrieved.")

    # Extract values for all 8 features and ensure that they're numeric
    features = {
        'ELEVATION': dem.reduceRegion(ee.Reducer.mean(), point, 30).get('elevation').getInfo() or 0,
        'SLOPE': slope.reduceRegion(ee.Reducer.mean(), point, 30).get('slope').getInfo() or 0,
        'ASPECT': aspect.reduceRegion(ee.Reducer.mean(), point, 30).get('aspect').getInfo() or 0,
        'TWI': twi.reduceRegion(ee.Reducer.mean(), point, 30).getInfo().get('constant') or 0,
        'SPI': spi.reduceRegion(ee.Reducer.mean(), point, 30).getInfo().get('constant') or 0,
        'NDVI': ndvi.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo() or 0,
        'RAINFALL': rainfall.reduceRegion(ee.Reducer.mean(), point, 30).get('precipitation').getInfo() or 0,
        'LANDUSE': landuse.reduceRegion(ee.Reducer.mode(), point, 30).getInfo().get('Map') or 0,
    }

    return features

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get latitude and longitude from the form
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])

            # Fetch features from Google Earth Engine
            earth_engine_features = get_earth_engine_features(latitude, longitude)

            # Log the fetched features
            fetched_features_log = f"Fetched Earth Engine features: {earth_engine_features}"

            # Extract features for model prediction (without LATITUDE and LONGITUDE)
            input_values = []
            for feature in feature_names:
                value = earth_engine_features.get(feature, 0)  # Default to 0 if feature is missing
                input_values.append(value)

            # Log the input values
            input_values_log = f"Input values for model: {input_values}"

            # Convert input values to a DataFrame (matches model input, excluding LATITUDE and LONGITUDE)
            input_data = pd.DataFrame([input_values], columns=feature_names)

            # Predict using the loaded model
            prediction = model.predict(input_data)[0]

            # Log the prediction
            model_prediction_log = f"Model prediction: {prediction}"

            # Map prediction (assuming '1' is High Risk and '0' is Low Risk)
            risk = "High Risk" if prediction == 1 else "Low Risk"

            # Render the index page with features, logs, and prediction
            return render_template('index.html', 
                                   earth_engine_features=earth_engine_features, 
                                   prediction=risk,
                                   fetched_features_log=fetched_features_log,
                                   input_values_log=input_values_log,
                                   model_prediction_log=model_prediction_log)

        except Exception as e:
            return f"An error occurred: {e}", 400

    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
