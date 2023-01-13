from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import numpy as np
import pandas as pd 
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import sys
import time
import tensorflow.keras as keras
import tensorflow as tf
import re
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import string
from flask_cors import CORS, cross_origin
import base64
#df = pd.read_excel('dataset excel.xlsx')

from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
# from keras.layers.experimental.preprocessing import Normalization
from keras.layers import ELU, PReLU, LeakyReLU
from keras.models import Model, Sequential
from keras.preprocessing import image 
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import keras.applications.mobilenet_v2 as mobilenetv2


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app, supports_credentials=True) 
from platform import python_version
print(python_version())

# Model saved with Keras model.save()
MODEL_PATH = 'model_Sample_New.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
     
    
    
    
    
    categories = {0: 'air-filter', 1: 'back-light', 2: 'chain-cover', 3: 'chain-spocket', 
              4: 'magnet-coil', 5: 'drum',
              6: 'gear-lever', 7: 'handle', 8: 'head', 9: 'head-light', 10: 'horn',
              11: 'jumps', 12: 'kick', 13: 'magnet-cover', 14: 'meter', 15: 'mudguard',
              16: 'piston', 17: 'rim', 18: 'seat', 19: 'side-cover', 
              20: 'silencer', 21: 'tanki'}
    #img = image.load_img(img_path, target_size=(224, 224))
    
    img = cv2.imread(img_path) # Error
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])

    model.predict(img)
    classes = np.argmax(model.predict(img), axis = -1)

     #print('Predicted : ', categories[classes[0]])
    # Preprocessing the image
   # x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
   # x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x, mode='caffe')

    #preds = model.predict(x)
    return categories[classes[0]]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    #res = 'image'
    N=7
    content = request.json
    
    
    image = content['image']
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
#    image = content['image']
    decoded_data= base64.b64decode((image))
    image_path = str(res)+'.jpeg'
    img_file = open(image_path, 'wb')
    img_file.write(decoded_data)   
    
    
    
    if request.method == 'POST':
        # Get the file from post request
       # image_64_encode = request.files
      #  f = base64.decodestring(image_64_encode) 
       
        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(
         #   basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)

        # Make prediction
        preds = model_predict(image_path, model)

                    # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

