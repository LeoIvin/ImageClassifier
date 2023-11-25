import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

st.set_page_config(layout="centered")



st.write("""
         # Simple Image Classifier
         Upload an image and the classifier will tell you what object it is.
         """)

uploaded_file = st.file_uploader("Choose an image...", type=None)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = img.resize((224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make a prediction
    preds = model.predict(img)
    

    # Decode the prediction
    pred_class = decode_predictions(preds, top=1)[0][0]
    st.write("Prediction: ", pred_class[1])
    st.write("Confidence: ", round(pred_class[2]*100, 2), "%")
