import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import requests

from tensorflow.keras.models import load_model

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import speech_recognition as sr
import av
import re

from streamlit_geolocation import streamlit_geolocation
from PIL import Image


# LOAD MODELS
# ==============================
crop_model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/crop_scaler.pkl")
label_encoder = joblib.load("models/crop_label.pkl")
MODEL_PATH = "models/disease_model.h5"

# ==============================
# HEADER
# ==============================

st.markdown(""" 
<h2 style="
text-align:center;
color:#FFF000;
font-weight: bold;
font-weight:700;">
🧑‍🌾🌱AI-Powered Smart Farming Assistant
</h2>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
text-align:center;
color:#000000;
font-weight: bold;
font-size:15px;">
Crop Recommendation|Leaf Disease Detection  & Fertilizer Suggestion|Soil Image-Based Analysis using gps
</p>
""", unsafe_allow_html=True)


# Background + UI Styling
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
}

/* Glass box */
.box {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 10px;
}

/* Input labels */
label {
    color: #ffffff !important;
    font-weight: bold;
}

/* Inputs */
.stNumberInput input {
    background-color: #1e1e1e;
    color: white;
    border-radius: 8px;
}

/* Button */
.stButton > button {
    background: linear-gradient(to right, #56ab2f, #a8e063);
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -----------Title---------------------

st.markdown("""
<h3 style="
color:#7FFF00;
font-weight:800;
font-size:25px;
text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
">
🌱 Crop Recommendation
</h3>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

# ROW 1
col1, col2 = st.columns(2)
with col1:
    N = st.number_input("🌿 Nitrogen (N) [0-140]", 0, 140, 0)
with col2:
    temp = st.number_input("🌡 Temperature (°C) [0-50]", 0, 50, 25)

# ROW 2
col1, col2 = st.columns(2)
with col1:
    P = st.number_input("🧪 Phosphorus (P) [0-145]", 0, 145, 0)
with col2:
    humidity = st.number_input("💧 Humidity (%) [0-100]", 0, 100, 50)

# ROW 3
col1, col2 = st.columns(2)
with col1:
    K = st.number_input("⚡ Potassium (K) [0-205]", 0, 205, 0)
with col2:
    ph = st.number_input("🧫 Soil pH [0-14]", 0.0, 14.0, 7.0)

# ROW 4
col1, col2 = st.columns(2)
with col1:
    st.empty()  # keeps alignment
with col2:
    rainfall = st.number_input("🌧 Rainfall (mm) [0-300]", 0, 300, 100)
# Button
if st.button("🌾 Recommend Crop"):
    input_data = [[N, P, K, temp, humidity, ph, rainfall]]
    input_scaled = scaler.transform(input_data)
    prediction = crop_model.predict(input_scaled)

    crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Recommended Crop: {crop}")

# ==============================
# WEATHER FUNCTION
# ==============================
API_KEY = "YOUR_OPENWEATHER_API_KEY"

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m,precipitation"
    
    response = requests.get(url)
    data = response.json()

    try:
        temperature = data["current_weather"]["temperature"]
        humidity = data["hourly"]["relativehumidity_2m"][0]
        rainfall = data["hourly"]["precipitation"][0]

        return temperature, humidity, rainfall
    except:
        return None

import geocoder

g = geocoder.ip('me')

# Crop Recommendation Function
def recommend_crop(N, P, K, temp, humidity, ph, rainfall):

    # Create input array (7 features)
    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    pred = crop_model.predict(data_scaled)

    # Convert label to crop name
    crop = label_encoder.inverse_transform(pred)[0]

    return crop

# ==============================
# DISEASE PREDICTION
# ==============================

import tensorflow as tf

disease_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


import joblib

class_indices = joblib.load("models/disease_classes.pkl")
disease_labels = list(class_indices.keys())

def predict_disease(image):

    img = cv2.imread(image)

    img = cv2.resize(img,(128,128))   # IMPORTANT

    img = img / 255.0

    img = np.reshape(img,(1,128,128,3))

    prediction = disease_model.predict(img)

    class_index = np.argmax(prediction)

    return class_names[class_index]


# ==============================
# GPS WEATHER SECTION
# ==============================
st.markdown(""" 
<h2 style="
text-align:LEFT;
font-size:25px;
color:#FFF000;
font-weight:700;
text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
">
📍Real-Time Weather using GPS
</h2>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Target geolocation container */
[data-testid="stHorizontalBlock"] {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

/* Fix spacing */
[data-testid="stHorizontalBlock"] > div {
    display: flex;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)



location = streamlit_geolocation()
gps_on = location and location.get("latitude")

if gps_on:
    st.markdown("🟢 Location ON")

if location and location.get("latitude"):

    lat = location["latitude"]
    lon = location["longitude"]

   
    st.success(f"📌 Latitude : {lat}")
    st.success(f"📌 Longitude : {lon}")

    weather = get_weather_data(lat, lon)

    if weather:
        temperature, humidity, rainfall = weather

        st.success(f"🌡 Temp: {temperature}°C")
        st.success(f"💧 Humidity: {humidity}%")
        st.success(f"🌧 Rainfall: {rainfall}mm")
else:
    st.markdown("  **Location OFF - Allow browser permission**")


st.markdown("""
<style>

/* Make location icon small */
iframe{
transform: scale(1.0);
}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# Load trained model
# ---------------------------
model = load_model("plant_disease_model.h5")
# ---------------------------
# Class names from dataset
# ---------------------------
class_names = [
"Pepper_bell__Bacterial_spot",
"Pepper_bell__healthy",
"Potato__Early_blight",
"Potato__healthy",
"Potato__Late_blight",
"Tomato_Bacterial_spot",
"Tomato_healthy"

]

# ---------------------------
# Page title
# ---------------------------

st.markdown(""" 
<h2 style="
text-align:LEFT;
color:#7FFF00;
font-weight:700;
font-size:25px;
text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
">
🍃Leaf Disease Detection
</h2>
""", unsafe_allow_html=True)
# --------------------------
# Upload image
# ---------------------------
st.markdown("""
<style>

/* Make uploader compact */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.7);
    padding: 10px;
    border-radius: 10px;
    min-height:100px;
}

/* Remove big drag box */
[data-testid="stFileUploaderDropzone"] {
    padding: 5px !important;
    border: none !important;
}

/* Hide extra text */
[data-testid="stFileUploaderDropzone"] div {
    font-size: 12px !important;
}

/* Browse button styling */
[data-testid="stFileUploader"] button {
    background-color: #6c757d;
    color: white;
    border-radius: 8px;
}

/* Reduce spacing */
.css-1cpxqw2 {
    margin-bottom: 0px !important;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Upload title */
[data-testid="stFileUploader"] label {
    color: #000000 !important;
    font-weight: 600;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)



uploaded_file = st.file_uploader(
"Upload Leaf Image",
type=["jpg","jpeg","png"]
)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image)

    img = np.array(image)
    img = cv2.resize(img,(128,128))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Detected Disease: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Fertilizer dictionary
    fertilizer_dict = {
        "Pepper__bell__Bacterial_spot":
        "Apply copper based bactericide and balanced NPK fertilizer.",

        "Pepper__bell__healthy":
        "Plant is healthy. Apply organic compost.",

        "Potato__Early_blight":
        "Use Mancozeb fungicide and apply potassium rich fertilizer.",

        "Potato__healthy":
        "Plant is healthy. Maintain soil nutrients with NPK fertilizer.",

        "Potato__Late_blight":
        "Use Metalaxyl fungicide and avoid excess watering.",

        "Tomato_Bacterial_spot":
        "Apply copper fungicide and calcium nitrate fertilizer.",

        "Tomato_healthy":
        "Plant is healthy. Apply organic manure."
    }

    fertilizer = fertilizer_dict.get(predicted_class, "General Organic Fertilizer")

    st.subheader("Recommended Fertilizer")
    st.success(fertilizer)
# ==============================
# LOAD NEW CROP MODEL (Soil + Weather + Season)
# ==============================
import datetime

crop_model = joblib.load("models/crop_model_new.pkl")
soil_encoder = joblib.load("models/soil_encoder.pkl")
season_encoder = joblib.load("models/season_encoder.pkl")
label_encoder = joblib.load("models/crop_label_new.pkl")

# ==============================
# LOAD SOIL CNN MODEL
# ==============================
soil_model = load_model("models/soil_model.h5", compile=False)

soil_classes = [
    'Black Soil',
    'Laterite Soil',
    'Clay Soil',
    'Red Soil',
    'Sandy Soil'
]

# ==============================
# SOIL PREDICTION
# ==============================
def predict_soil(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = soil_model.predict(img)
    soil_type = soil_classes[np.argmax(prediction)]

    return soil_type
# ==============================
# AUTO SEASON DETECTION
# ==============================
def get_current_season():
    month = datetime.datetime.now().month

    if month in [6,7,8,9,10]:
        return "Rainy sesaon"

    elif month in [11,12,1,2]:
        return "winter season"
    else:
        return "summer season"
    
    # ==============================
# NEW CROP RECOMMENDATION
# ==============================
def recommend_crop(soil_type, temp, humidity, rainfall):

    season = get_current_season()

    soil_encoded = soil_encoder.transform([soil_type])[0]
    season_encoded = season_encoder.transform([season])[0]

    data = np.array([[soil_encoded, temp, humidity, rainfall, season_encoded]])

    pred = crop_model.predict(data)

    crop_name = label_encoder.inverse_transform(pred)[0]

    return crop_name, season


# ==============================
# SOIL IMAGE UPLOAD
# ==============================
st.markdown(""" 
<h2 style="
text-align:LEFT;
color:#FFF000;
font-weight:700;
font-size:25px;
text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
">
🟫Soil Analysis & Crop Recommendation
</h2>
""", unsafe_allow_html=True)

soil_image = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])

detected_soil = None


def calculate_soil_health(soil_type, humidity, rainfall):

    soil_quality = {
        "Black Soil": 80,
        "Red Soil": 65,
        "Clay Soil": 70,
        "Sandy Soil": 45,
        "Laterite Soil": 55
    }

    base = soil_quality.get(soil_type,50)

    humidity_score = humidity * 0.2
    rainfall_score = rainfall * 0.1

    health = base + humidity_score + rainfall_score

    if health > 100:
        health = 100

    return round(health,2)

def healthy_crops(soil_type):

    crops = {

    "Black Soil": ["Cotton","Soybean","Groundnut"],

    "Red Soil": ["Millet","Potato","Groundnut"],

    "Clay Soil": ["Rice","Broccoli","Cabbage"],

    "Sandy Soil": ["Watermelon","Peanut","Carrot"],

    "Laterite Soil": ["Cashew","Tea","Coffee"]

    }

    return crops.get(soil_type,["Rice"])
def soil_treatment(soil_type):

    treatment = {

    "Black Soil":
    "Add organic compost and improve drainage",

    "Red Soil":
    "Add nitrogen fertilizer and organic manure",

    "Clay Soil":
    "Improve drainage and mix sand",

    "Sandy Soil":
    "Add compost and increase irrigation",

    "Laterite Soil":
    "Add lime and organic fertilizer"

    }

    return treatment.get(soil_type,"Use organic fertilizers")


if soil_image:
    detected_soil = predict_soil(soil_image)
    st.success(f"🟫 Detected Soil Type: {detected_soil}")
    
    if detected_soil and weather:
        temperature, humidity, rainfall = weather
        crop, season = recommend_crop(
        detected_soil,
        temperature,
        humidity,
        rainfall
    )

    st.success(f"🌦 Season: {season}")
    st.success(f"🌾 Recommended Crops: {crop}")
    

st.markdown(""" 
<h2 style="
text-align:LEFT;
color:#FFF000;
font-size:25px;
font-weight:700;
text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
">
🧪Soil Health Monitoring System
</h2>
""", unsafe_allow_html=True)

 
soil_image = st.file_uploader("Upload Soil Image")

if soil_image:

    soil_type = predict_soil(soil_image)

    st.success("Detected Soil Type: " + soil_type)

    health = calculate_soil_health(soil_type, humidity, rainfall)

    st.subheader("Soil Health: " + str(health) + "%")

    if health >= 60:

        st.success("Soil is Healthy")

        crops = healthy_crops(soil_type)

        st.subheader("Recommended Crops")

        for c in crops:
            st.write("•", c)

    else:

        st.error("Soil is Unhealthy")

        treatment = soil_treatment(soil_type)

        st.subheader("Treatment")

        st.write(treatment)

        crops = healthy_crops(soil_type)

        st.subheader("Suggested Crops")

        for c in crops:
            st.write("•", c)

# ---------- CUSTOM UI STYLE ----------
st.markdown("""
<style>

h1 {
    text-align: center;
    color: #1b5e20;
    font-size: 50px;
}

h2, h3 {
    color: #000000;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stFileUploader {
    background-color: rgba(255,255,255,0.8);
    padding: 10px;
    border-radius: 10px;
}

.result-box {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)
    
# ==============================
# FOOTER
# ==============================
st.markdown("""
<style>

/* Footer Styling */
.footer {
    text-align: center;
    font-weight: 700;
    font-size: 18px;
    color: #A7F3D0;  /* light green */
    margin-top: 100px;
    
    padding: 10px;
}

/* Optional: make it more attractive */
.footer {
    background: rgba(0,0,0,0.4);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
   
st.markdown(
    '<div class="footer">🌾 AI-based Smart Crop Advisory System</div>',
    unsafe_allow_html=True
)
