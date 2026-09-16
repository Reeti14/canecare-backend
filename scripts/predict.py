from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your vercel URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'canecare', 'canecare_effnetb3.keras')
CLASS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'canecare', 'class_names.json')

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)

if any(isinstance(v, int) for v in class_names.values()):
    idx_to_class = {v: k for k, v in class_names.items()}
else:
    idx_to_class = {int(k): v for k, v in class_names.items()}

@app.get("/")
def health():
    return {"status": "ok", "model": "canecare-effnetb3"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((300, 300), Image.Resampling.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    output = model.predict(img_array, verbose=0)[0]
    
    probabilities = [
        {"label": idx_to_class.get(idx, f"Class {idx}"), "probability": float(prob)}
        for idx, prob in enumerate(output)
    ]
    probabilities.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        "predictions": probabilities,
        "top": probabilities[0]
    }