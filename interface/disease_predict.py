# pyright: reportMissingImports=false

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import json

# Load model
model = load_model('models/leaf_disease_model2.h5')

# Load class labels
with open('models/class_indices2.json') as f:
    class_to_index = json.load(f)

# Invert dictionary: index (as string) -> class name
class_names = {str(v): k for k, v in class_to_index.items()}

def predict(img_path):
    img = Image.open(img_path).convert('RGB') 
    img = img.resize((128, 128)) 
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    
    
    predictions = model.predict(img_array)
    print("Raw predictions:", predictions)
    print(model.summary())
    print("Input image shape:", img_array.shape)
    print("Input pixel range:", img_array.min(), img_array.max())



    predicted_index = str(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    
    return predicted_class
