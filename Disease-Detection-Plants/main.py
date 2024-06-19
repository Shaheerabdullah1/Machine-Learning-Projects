import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import joblib

app = FastAPI()

# Model loading functions
def load_wheat_model():
    return load_model('Wheat-Model.h5')

def load_corn_model():
    return load_model('Corn-Model.h5')

def load_potato_model():
    return load_model('Potato-Model.h5')

def load_sugarcane_model():
    return load_model('Sugarcane-Model.h5')

def load_rice_model():
    return load_model('Rice-Model.h5')

def load_tomato_model():
    return load_model('Tomato-Model.h5')

# Load the models
wheat_model = load_wheat_model()
corn_model = load_corn_model()
potato_model = load_potato_model()
sugarcane_model = load_sugarcane_model()
rice_model = load_rice_model()
tomato_model = load_tomato_model()

# Helper function to process and predict image
def predict_image(model, image_file, class_names, confidence_threshold=0.8, image_size=(299, 299)):
    img = cv2.imread(image_file)
    if img is None:
        return "Invalid Image", None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = float(predictions[0][class_idx])

    if confidence < confidence_threshold:
        return "Invalid Image", confidence

    predicted_class = class_names[class_idx]
    return predicted_class, confidence

# API endpoint for Wheat
@app.post("/Wheat")
async def predict_wheat(file: UploadFile = File(...)):
    class_names = {
        0: 'Brown Rust',
        1: 'Healthy',
        2: 'Yellow Rust',
        3: 'Septoria'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(wheat_model, file_location, class_names)

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.8 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

# API endpoint for Corn
@app.post("/Corn")
async def predict_corn(file: UploadFile = File(...)):
    class_names = {
        0: 'Blight',
        1: 'Common Rust',
        2: 'Gray Leaf Spot',
        3: 'Healthy'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(corn_model, file_location, class_names)

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.8 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

# API endpoint for Potato
@app.post("/Potato")
async def predict_potato(file: UploadFile = File(...)):
    class_names = {
        0: 'Early Blight',
        1: 'Healthy',
        2: 'Late Blight'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(potato_model, file_location, class_names)

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.8 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

# API endpoint for Sugarcane
@app.post("/Sugarcane")
async def predict_sugarcane(file: UploadFile = File(...)):
    class_names = {
        0: 'Healthy',
        1: 'Mosaic',
        2: 'Red Rot',
        3: 'Rust',
        4: 'Yellow Spot'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(sugarcane_model, file_location, class_names)

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.8 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

# API endpoint for Rice
@app.post("/Rice")
async def predict_rice(file: UploadFile = File(...)):
    class_names = {
        0: 'Bacterial Leaf Blight',
        1: 'Brown Spot',
        2: 'Healthy',
        3: 'Leaf Blast',
        4: 'Leaf Scald',
        5: 'Narrow Brown Spot'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(rice_model, file_location, class_names)

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.8 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

# API endpoint for Tomato
@app.post("/Tomato")
async def predict_tomato(file: UploadFile = File(...)):
    class_names = {
        0: 'Bacterial_spot',
        1: 'Early_blight',
        2: 'Late_blight',
        3: 'Leaf_Mold',
        4: 'Septoria_leaf_spot',
        5: 'Spider_mites Two-spotted_spider_mite',
        6: 'Target_Spot',
        7: 'Mosaic_virus',
        8: 'Yellow_Leaf_Curl_Virus',
        9: 'Healthy'
    }

    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        predicted_class, confidence = predict_image(tomato_model, file_location, class_names, confidence_threshold=0.99, image_size=(224, 224))

        os.remove(file_location)

        return {"result": f"Prediction: {predicted_class}" if confidence >= 0.99 else "Prediction: Invalid Image"}
    else:
        raise HTTPException(status_code=400, detail='Invalid file type')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and label encoder
model_path = 'crop_recommendation_model.joblib'
label_encoder_path = 'label_encoder.joblib'
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)


class CropFeatures(BaseModel):
    N: str
    P: str
    K: str
    temperature: str
    humidity: str
    ph: str
    rainfall: str


def convert_to_float(value):
    try:
        logging.debug(f"Converting value {value} to float")
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid input: {value} is not a valid number")


def get_top_3_alternatives(features, model, label_encoder):
    probabilities = model.predict_proba([features])[0]
    top_4_indices = np.argsort(probabilities)[-4:][::-1]  # Get top 4 because the first one will be the predicted crop
    top_4_crops = label_encoder.inverse_transform(top_4_indices)
    return top_4_crops[1:].tolist()  # Return top 3 alternatives, excluding the first one which is the predicted crop


def predict_crop_and_alternatives(features):
    predicted_crop_index = model.predict([features])[0]
    predicted_crop = label_encoder.inverse_transform([predicted_crop_index])[0]
    top_3_alternatives = get_top_3_alternatives(features, model, label_encoder)
    return predicted_crop, top_3_alternatives


@app.post("/predict/")
async def predict_crop(features: CropFeatures):
    try:
        logging.debug(f"Received features: {features}")
        feature_list = [
            convert_to_float(features.N),
            convert_to_float(features.P),
            convert_to_float(features.K),
            convert_to_float(features.temperature),
            convert_to_float(features.humidity),
            convert_to_float(features.ph),
            convert_to_float(features.rainfall),
        ]
        logging.debug(f"Feature list: {feature_list}")
        predicted_crop, top_3_alternatives = predict_crop_and_alternatives(feature_list)
        return {
            "Best-Recommended-Crop": predicted_crop,
            "Top-3-Alternatives": top_3_alternatives
        }
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

