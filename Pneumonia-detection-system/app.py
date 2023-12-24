import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model('Model.h5')  # Load your trained model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Preprocess the image
        img_arr = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (150, 150))
        resized_arr = resized_arr / 255.0
        input_data = resized_arr.reshape(1, 150, 150, 1)

        # Make prediction
        prediction = model.predict(input_data)[0][0]
        
        os.remove(filename)  # Remove the uploaded file

        result = "Normal" if prediction > 0.5 else "Pneumonia"
        return render_template('index.html', result=result)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
