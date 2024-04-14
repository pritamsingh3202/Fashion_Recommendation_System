import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
import streamlit as st
import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


model = ResNet50(weights= 'imagenet', include_top= False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([ model, GlobalMaxPool2D()])

st.title('FASHION RECOMMENDER SYSTEM')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())
        return 1
    except:
        return 0
def features_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
# file upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = features_extraction(os.path.join("uploads", uploaded_file.name), model)
        st.text(features)

        indices = recommend(features, feature_list)

        col1, col2, col3, col4, col5, col6, col7 = st.beta_columns(7)

        with col1:
            st.image(filenames[indices[0][0]])
    else:
        st.header("Some error occurred in file upload")

