"""
NOTE: To run this program as a server, start this program in the terminal
using the following command:

   uvicorn cloud_web_app:app

This will start the server and print out the link to the port that it is running on.
Copy and paste this link into the fetch command in the JavaScript file if the link has changed

To install all dependencies if running in a new pycharm project, run the following command:
   pip install tensorflow numpy matplotlib seaborn scikit-learn pillow fastapi uvicorn python-multipart
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import io

model_path = "cloud_identifier_mobilenet.keras"
model = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Cirrus', 'Cumulonimbus', 'Cumulus', 'Stratus']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/") :
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except:
        raise HTTPException(status_code=500, detail="Invalid image file.")

    image = image.convert("RGB")
    image = image.resize((400, 400))

    np_image = np.array(image)
    input_image_array = np.expand_dims(np_image, axis=0)

    predictions = model.predict(input_image_array, verbose=0)[0]
    prediction_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[prediction_index]
    confidence = float(predictions[prediction_index]*100)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predictions":{
            CLASS_NAMES[i]: round(float(predictions[i]*100), 2)
            for i in range(len(CLASS_NAMES))
            },
    }








