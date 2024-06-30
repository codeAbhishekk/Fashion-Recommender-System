import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list= np.array(pickle.load(open('embeddings.pkl','rb')))
# print(feature_list)
# print(np.array(feature_list).shape)

filenames= pickle.load(open('filenames.pkl','rb'))

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.build(input_shape=(None, 224, 224, 3))

img_path = 'sample/jersey.jpg'
img =image.load_img(img_path, target_size=(224, 224))  # Corrected function
img_array = tf.keras.utils.img_to_array(img)  # Corrected function
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)



neighbors=NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')

neighbors.fit(feature_list)

distances, indices=neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)

