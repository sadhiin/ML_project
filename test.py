import numpy as np
from numpy.linalg import norm
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import cv2
import tensorflow

# print(cv2.__version__)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# # Importing the feature list and the file names
# feature_list = np.array(pickle.load(open('features.pkl', "rb")))
# filenames_list = pickle.load(open('filenames.pkl', 'rb'))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(
    224, 224, 3))  # A model is that trained on imagenet

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

# # Location of the Test image
# test_image_path = "../Datasets/Test/1662.jpg"

# # Feature extraction for the test image
# test_img = image.load_img(test_image_path, target_size=(224, 224))
# test_img_array = image.img_to_array(test_img)

# expanded_test_image_array = np.expand_dims(test_img_array, axis=0)
# preprocessed_test_img = preprocess_input(expanded_test_image_array)

# test_outcome = model.predict(preprocessed_test_img).flatten()
# normalized_test_outcome = test_outcome/norm(test_outcome)

# # End of Feature extraction


# # KNN implementation [Closest images]

# all_neighbor = NearestNeighbors(
#     n_neighbors=6, algorithm='brute', metric='euclidean')
# all_neighbor.fit(feature_list)

# # distances, indexs = all_neighbor.kneighbors([normalized_test_outcome])

# # print(indexs)

# # for id in indexs[0]:
# #     temp_image = cv2.imread(filenames_list[id])
# #     # cv2.imshow('Output',cv2.resize(temp_image,(512,512)))
# #     cv2.imshow('Output',temp_image)

# #     cv2.waitKey(0)

# distence, indices = all_neighbor.kneighbors([normalized_test_outcome])

# print(indices)

# # 🆘🆘🆘🆘🆘🆘
# # for file in indices[0][1:6]:
# #     print(filenames_list[file])
# #     tmp_img = cv2.imread(filenames_list[file])
# #     # tmp_img = cv2.resize(tmp_img,(200,200))
# #     cv2.imshow("output", tmp_img)
# #     cv2.waitKey(0)


# # End of presention
