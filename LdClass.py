# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:49:56 2022

@author: sp7012
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import load_model
from PIL import Image
model = load_model('model_saved.h5')

image = load_img('v_data/test/54.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])

if   label[0][0]<0.5:

    im = Image.open("v_data/train/cars/5.jpg")

#show image
    im.show()
elif label[0][0]>=0.5:

    im = Image.open("v_data/train/planes/2.jpg")

#show image
    im.show()