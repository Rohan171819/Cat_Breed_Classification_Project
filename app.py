import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Data Science Project", layout="wide")
st.title("Cat Breed Classification")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# Load the model
model = tf.keras.models.load_model('Cat Breed Classifier.h5')
prediction_button = st.button("Predict")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    

    if prediction_button:
        # Convert the uploaded image to a NumPy array
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        image_array = tf.expand_dims(image_array, axis=0)  # Add a batch dimension

        # Make predictions using the model
        predictions = model.predict(image_array)

        # Get the class label corresponding to the predicted index
        class_labels = ['bengal', 'domestic_shorthair', 'maine_coon', 'ragdoll', 'siamese']
        predicted_label = class_labels[np.argmax(predictions)]

        # Display the prediction
        st.success("Prediction: {}".format(predicted_label))
else:
    st.info("Please upload an image file to make a prediction.")