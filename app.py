import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the ResNet50 model with ImageNet weights, excluding the top layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.build(input_shape=(None, 224, 224, 3))

def extract_features(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))  # Corrected function
    img_array = tf.keras.utils.img_to_array(img)  # Corrected function
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Collect filenames from the images directory
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Extract features for each image
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# print(np.array(feature_list).shape)

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))