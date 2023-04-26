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
            print(predictions)
            
            if predictions[0][0] > predictions[0][1]:
                prediction = "Cancer"
            else:
                prediction = "No Cancer"
                
            result_dic = {
                'image' : image_path,
                'prediction' : prediction
            }

        return render_template('pages/index.html', results = result_dic)