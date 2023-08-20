import streamlit as st
import os
import cv2
import zipfile
import matplotlib.pyplot as plt
from numpy.linalg import norm
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import requests
import urllib.request
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.layers import GlobalMaxPooling2D
from io import BytesIO

file_name=np.load("C:\\Users\\lenovo\\Downloads\\filename_list.npy")
final_list=np.load("C:\\Users\\lenovo\\Downloads\\feature_list_50k.npy")

img_paths=[]
for i in file_name:
    img_paths.append('D:\\GRID\\'+os.path.split(os.path.split(i)[0])[1]+'/'+os.path.split(i)[1])


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(640,640,3))
base_model.trainable=False
model=keras.Sequential([
    base_model, 
    GlobalMaxPooling2D(),
])

st.title("Similar Items you may like:")

img_url='https://image.hm.com/assets/hm/c8/c2/c8c2a16dbdd0cbd055a685a39a22a90f9c1c68e4.jpg?imwidth=396'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
response = requests.get(img_url)
img=Image.open(BytesIO(response.content))
img=img.resize((400,400))
st.image(img)

def extract_featuresknn(image_url,model):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    req = urllib.request.Request(image_url, headers=headers)
    response = urllib.request.urlopen(req)
    img = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image_array = cv2.imdecode(img, cv2.IMREAD_COLOR)
    a=cv2.resize(image_array, (640,640))
    a=np.expand_dims(a,axis=0)  #model takes 4d tensor/image as input
    pre=preprocess_input(a)
    result=model.predict(pre).flatten()
    normalized=result/norm(result)
    return normalized.reshape(1,-1)

def recommend(img_url,final_list):
    neighbors=NearestNeighbors(n_neighbors=10,algorithm='brute', metric='euclidean')
    neighbors.fit(final_list)
    distance, indices=neighbors.kneighbors(extract_featuresknn(img_url,model))
    return indices

indices=recommend(img_url,final_list)

l=list(indices.reshape(-1))[1:]

num_columns = 3
num_rows = len(l) // num_columns + (len(l) % num_columns != 0)

for row in range(num_rows):
    cols=st.columns(num_columns)
    for col_index, col in enumerate(cols):
        img_index=row*num_columns+col_index
        with col:
            img=Image.open(img_paths[l[img_index]])
            st.image(img)
            st.write(img_index)

# col = st.columns(len(l))
# zip_file_path = "D:\GRID\images.zip" 
# # for i in l:

# for i,co in enumerate(col):
#     with co:



# for i in col:
#     st.image(file_name[])









