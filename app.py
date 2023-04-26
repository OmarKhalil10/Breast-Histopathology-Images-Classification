import os
from flask import Flask, request, render_template, redirect, abort, jsonify, flash, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam
from flask import Flask, send_from_directory
from keras.preprocessing import image
from PIL import Image


# configure allowed extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__)
    app.config.from_pyfile('settings.py')
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization,true')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE, OPTIONS')
        return response
    
    @app.route('/')
    def index():
        return render_template('pages/index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():


        # check if the post request has the file part
        if request.method == 'POST':
            if 'image' not in request.files:
                flash('No file part')

            file = request.files['image']
       
            if file.filename == '':
                    flash('No File Selected')

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
            model =  tf.keras.models.load_model('.\\model.h5')
            
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(path, target_size=(224, 224))
            expand_input = np.expand_dims(img,axis=0)
            input_data = np.array(expand_input)
            input_data = input_data/255

            pred = model.predict(input_data)

            print(pred)

            if pred >= 0.5:
                prediction = "Cancer"
            else:
                prediction = "No Cancer"
            
            result_dic = {
                'image' : path,
                'prediction' : prediction
            }

        return render_template('pages/index.html', results = result_dic)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4040, debug=True)
