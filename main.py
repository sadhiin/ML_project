import os
from PIL import Image
from pyparsing import col
import streamlit as st
import numpy as np
from numpy.linalg import norm
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import cv2
import tensorflow


from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Importing the feature list and the file names
feature_list = np.array(pickle.load(open('features.pkl', "rb")))
filenames_list = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(
    224, 224, 3))  # A model is that trained on imagenet

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# Feature extraction for the test image
def feature_extractor (file_path,model):
    test_img = image.load_img(test_image_path, target_size=(224, 224))
    test_img_array = image.img_to_array(test_img)

    expanded_test_image_array = np.expand_dims(test_img_array, axis=0)
    preprocessed_test_img = preprocess_input(expanded_test_image_array)

    test_outcome = model.predict(preprocessed_test_img).flatten()
    normalized_test_outcome = test_outcome/norm(test_outcome)
    return normalized_test_outcome
# End of Feature extraction


def recomender(features,feature_list):
    all_neighbor = NearestNeighbors(
    n_neighbors=6, algorithm='auto', metric='euclidean')
    all_neighbor.fit(feature_list)

    distances, indexs = all_neighbor.kneighbors([features])

    return indexs


def save_uploaded_files(up_file):
    try:
        with open (os.path.join('uploaded_img',up_file.name),'wb') as f:
            f.write(up_file.getbuffer())
        return 1
    except:
        return 0


# Creating the streamlit app
st.title("Fashion Recomendation System Using CNN")


upload_file = st.file_uploader('Chose an Image .jpg') # Uploader

if upload_file is not None:
    test_image_path = os.path.join("uploaded_img/",upload_file.name)

# print(st.__version__)
#steps....
# 1. file upload ->save


# If the upload done properly.....
if upload_file is not None:
    if save_uploaded_files(upload_file):
        # display the image
        display_img = Image.open(upload_file)
        st.image(display_img)
        uploaded_features = feature_extractor(test_image_path,model)

        st.text(uploaded_features)
        idxs = recomender(uploaded_features, feature_list)
        col1,col2,col3,col4,col5,col6 = st.columns(6)

        with col1:
            st.image(filenames_list[idxs[0][0]])
        with col2:
            st.image(filenames_list[idxs[0][1]])
        with col3:
            st.image(filenames_list[idxs[0][2]])
        with col4:
            st.image(filenames_list[idxs[0][3]])
        with col5:
            st.image(filenames_list[idxs[0][4]])
        with col6:
            st.image(filenames_list[idxs[0][5]])
    else:
        st.header('Error occured in file upload')

# 2. load file -> feature extraction
# 3. Recomandaton
# 4. show the files