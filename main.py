import os

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras.applications.vgg16 import VGG16,  preprocess_input
# from keras.src.applications.vgg16 import VGG16
# from keras.src.applications.vgg16 import preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from PIL import Image

import pickle

Image.LOAD_TRUNCATED_IMAGES = True






model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# print(model.summary())

def features_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

filenames = []

for file in os.listdir('Images'):
    filenames.append(os.path.join('Images', file))

# print(len(filenames))
# print(filenames[0:5])
# print(os.listdir('Images'))

features_list =[]

for file in tqdm(filenames):
    try:
        features_list.append(features_extraction(file, model))
    except OSError:
        print(f"Skipping corrupted image {file}")

# print(np.array(features_list).shape)

pickle.dump(features_list, open('embedding.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
