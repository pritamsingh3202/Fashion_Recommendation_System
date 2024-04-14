import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# import os
import tensorflow
from sklearn.neighbors import NearestNeighbors
import cv2



model = ResNet50(weights= 'imagenet', include_top= False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


img = image.load_img('Sample images/9832.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:7]:
    team_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(team_img, (512, 512)))
    cv2.waitKey(0)
