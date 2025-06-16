import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf


import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report
from fastapi import FastAPI

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize FastAPI
app = FastAPI()

import os
import urllib.request
from tensorflow.keras.models import load_model

model_path = 'furniture.h5'
model_url = 'https://drive.google.com/uc?export=download&id=1lfU7i93OER64RQpxGGpa4ADH2QhY6rbe'

# Only download if the model file is missing
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    urllib.request.urlretrieve(model_url, model_path)

# Now load the model
model = load_model(model_path)



# Load trained furniture CNN model
class_names = ['bed', 'chair', 'sofa', 'swivelchair']

def generate_classification_report():
    # Your actual classification report data from Colab
    report_dict = {
        'bed': {'precision': 0.78, 'recall': 0.85, 'f1-score': 0.82, 'support': 200},
        'chair': {'precision': 0.88, 'recall': 0.79, 'f1-score': 0.83, 'support': 200},
        'sofa': {'precision': 0.73, 'recall': 0.85, 'f1-score': 0.79, 'support': 200},
        'swivelchair': {'precision': 0.89, 'recall': 0.92, 'f1-score': 0.90, 'support': 200},
        'accuracy': {'precision': 0.82, 'recall': 0.82, 'f1-score': 0.82, 'support': 1},
        'macro avg': {'precision': 0.83, 'recall': 0.79, 'f1-score': 0.80, 'support': 889},
        'weighted avg': {'precision': 0.83, 'recall': 0.82, 'f1-score': 0.82, 'support': 889}
    }

    # Convert to template format
    report_data = {
        'metrics': list(report_dict.keys()),
        'precision': [v['precision'] for v in report_dict.values()],
        'recall': [v['recall'] for v in report_dict.values()],
        'f1-score': [v['f1-score'] for v in report_dict.values()],
        'support': [v['support'] for v in report_dict.values()]
    }
    
    return report_data

def generate_confusion_matrix():
    # Confusion matrix values from your image
    confusion_matrix = {
        'class_names': ['bed', 'chair', 'sofa', 'swivelchair'],
        'matrix': [
            [174, 5, 21, 4, 4],    # bed
            [6, 156, 3, 3, 2],     # chair
            [9, 5, 180, 1, 2],     # sofa
            [0, 0, 0, 70, 0],      # swivelchair
        ]
    }
    return confusion_matrix

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    result = None
    report_data = None
    confusion_matrix = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image

            # Load image as RGB
            img = Image.open(filepath).convert('RGB')
            img = img.resize((180, 180))
            img = np.array(img)

            # Match training preprocessing: RGB → grayscale
            img = tf.image.rgb_to_grayscale(img)
            img_array = img.numpy().astype('float32')  # Remove division by 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 1)



            # Predict
            prediction = model.predict(img_array)
            percentages = (prediction[0] * 100).round(2).tolist()
            predicted_index = int(np.argmax(prediction[0]))
            predicted_label = class_names[predicted_index]
            confidence = percentages[predicted_index]

            # Prepare result
            result = {
                'label': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'percentages': percentages
            }
            
            # Generate reports
            report_data = generate_classification_report()
            confusion_matrix = generate_confusion_matrix()

    return render_template(
        'index.html',
        filename=filename,
        result=result,
        class_names=class_names,
        report_data=report_data,
        confusion_matrix=confusion_matrix
    )

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)