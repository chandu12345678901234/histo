# app.py
import os
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Function to load and preprocess images
def load_and_preprocess_images(images_dir, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    image_files = os.listdir(images_dir)
    X_train = np.zeros((len(image_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    missing_count = 0
    
    for n, image_file in enumerate(image_files):
        path = os.path.join(images_dir, image_file)
        try:
            img = imread(path)
            img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_train[n] = img
        except Exception as e:
            print(f"Problem with {path}: {str(e)}")
            missing_count += 1
    
    X_train = X_train.astype('float32') / 255.
    print("Total missing: " + str(missing_count))
    return X_train

# Load images
images_dir = '/content/extracted_project'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
X_train = load_and_preprocess_images(images_dir, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Streamlit app
st.title("Image Viewer")

# Select image from dropdown
selected_image_index = st.selectbox("Select an image:", list(range(len(X_train))))

# Display selected image
st.image(X_train[selected_image_index], caption=f"Image {selected_image_index + 1}", use_column_width=True)

# Display original images using matplotlib (as Streamlit does not support grayscale images directly)
st.subheader("Original Images (Matplotlib)")

num_images_to_display = min(5, len(X_train))
fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 5))

for i in range(num_images_to_display):
    img = X_train[i]
    axs[i].imshow(img)
    axs[i].axis('off')

st.pyplot(fig)

# Display a specific image using OpenCV (as Streamlit supports only RGB images)
st.subheader("Display a specific image using OpenCV")

specific_image_index = st.slider("Choose an index to display:", 0, len(X_train) - 1, 0)

# Convert grayscale image to RGB using OpenCV
specific_image = cv2.cvtColor((X_train[specific_image_index] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
st.image(specific_image, caption=f"Specific Image {specific_image_index + 1}", use_column_width=True)
