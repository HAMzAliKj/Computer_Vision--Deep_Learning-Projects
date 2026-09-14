import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import io


app = FastAPI()

# Load model once when FastAPI starts
model = load_model("model/Hamza123.keras")


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Open image
    img = image.load_img(
        io.BytesIO(contents),
        target_size=(224, 224)
    )

    # Convert image to array
    x = image.img_to_array(img)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # VGG16 preprocessing
    x = preprocess_input(x)

    # Prediction
    
    prediction = model.predict(x)

    result = int(np.argmax(prediction, axis=1)[0])

    classes = {
        0: "Oblique",
        1: "Spiral"
    }

    predicted_class = classes[result]

    confidence = float(prediction[0][result])

    return {
        "prediction": result,
        "class_name": predicted_class,
        "confidence": confidence
    }