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
                
            # load json and create model
            json_file = open('_model_.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("_model_.h5")
            print("Loaded model from disk")
            
            # evaluate loaded model on test data
            opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(image_path, target_size=(100,100))
            img=np.array(img)
            print('po array = {}'.format(img.shape))
            img = np.true_divide(img, 255)
            img = img.reshape(-1,100, 100,3)
            print(type(img), img.shape)
            predictions = model.predict(img)
            print(model)
            predictions_c = model.predict(img)
            print(predictions, predictions_c)
            model.predict(img)
            print(predictions_c)

            classes = {'TRAIN': ['Non Malignant (No Cancer)','Malignant'],
                    'TEST': ['Non Malignant (No Cancer)','Malignant']}

            # get the index of the class with the highest prediction probability
            predicted_class_idx = predictions_c.argmax()

            # use the index to get the corresponding class label from the 'TRAIN' category
            predicted_class = classes['TRAIN'][predicted_class_idx]
            
            prediction = predicted_class.lower()

            result_dic = {
                'image' : image_path,
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
