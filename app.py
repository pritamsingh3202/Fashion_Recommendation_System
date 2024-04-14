import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os
from tqdm import tqdm
# from PIL import Image
import pickle


model = ResNet50(weights= 'imagenet', include_top= False, input_shape=(224,224,3))
model.trainable= False

model= tensorflow.keras.Sequential([model,GlobalMaxPool2D()])

print(model.summary())

def extract_features(img_path,model):
    img= image.load_img(img_path, target_size=(224,224))
    img_array= image.img_to_array(img)
    expanded_img_array= np.expand_dims(img_array, axis=0)
    preprocess_img= preprocess_input(expanded_img_array)
    result= model.predict(preprocess_img).flatten()
    normalized_result= result/norm(result)

    return normalized_result

filenames= []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# print(len(filenames))
# print(filenames[0:10])

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# print(np.array(feature_list).shape)

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

