import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile


# Function to load trained model and predict class from input image
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("My_Trained_Model.keras")  # load saved model
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))  # resize image
    x = tf.keras.utils.img_to_array(img)  # convert image to array
    x = np.expand_dims(x, axis=0)  # add batch dimension
    x = preprocess_input(x)  # preprocess for MobileNetV3
    predictions = model.predict(x)  # get prediction probabilities
    return np.argmax(predictions)  # return index of highest probability


# Sidebar UI for navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])


# Home Page UI
if app_mode == "Home":
    st.markdown("""
## OCT Retinal Analysis Platform

Welcome to the Retinal OCT Analysis Platform.

OCT provides high-resolution retinal images to detect diseases like CNV, DME, and AMD.

### Features
- Automated Image Analysis
- Disease Classification: Normal, CNV, DME, Drusen
- Simple Upload & Prediction
    """)


# About Page UI
elif app_mode == "About":
    st.header("About")
    st.markdown("""
Retinal OCT captures cross-sectional images of the retina.

Dataset:
- 84,495 images
- Categories: NORMAL, CNV, DME, DRUSEN
- Organized into train/test/val
    """)


# Disease Identification Page
elif app_mode == "Disease Identification":
    st.header("Retinal OCT Analysis")

    # Upload image from user
    test_image = st.file_uploader("Upload your Image:")

    # Save uploaded image temporarily for model input
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name

    # Run prediction when button is clicked
    if st.button("Predict") and test_image is not None:
        with st.spinner("Please Wait.."):
            result_index = model_prediction(temp_file_path)  # get prediction index
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']  # class labels

        # Display prediction result
        st.success(f"Model predicts: {class_name[result_index]}")

        # Show additional info based on prediction
        with st.expander("Learn More"):
            if result_index == 0:
                st.write("CNV detected")
                st.image(test_image)
                st.markdown(cnv)

            elif result_index == 1:
                st.write("DME detected")
                st.image(test_image)
                st.markdown(dme)

            elif result_index == 2:
                st.write("Drusen detected")
                st.image(test_image)
                st.markdown(drusen)

            elif result_index == 3:
                st.write("Normal retina")
                st.image(test_image)
                st.markdown(normal)