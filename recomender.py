# import numpy as np
# from numpy.linalg import norm
# import seaborn as sns
# import os

# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

# model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) # A model is that trained on imagenet

# model.trainable = False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# # print(model.summary())

# #feature extraction function
# def feature_extractor(image_path,my_model):
#     img = image.load_img(image_path,target_size=(224,224))
#     img_array = image.img_to_array(img)
#     expanded_image_array = np.expand_dims(img_array,axis=0)
#     preprocessed_img = preprocess_input(expanded_image_array)
#     outcome = my_model.predict(preprocessed_img).flatten()
#     normalized_outcome = outcome / norm(outcome)
#     return normalized_outcome


# filenames = []
# datasets_directory = ""

# for file in os.listdir(datasets_directory):
#     filenames.append(os.path.join(datasets_directory, file))