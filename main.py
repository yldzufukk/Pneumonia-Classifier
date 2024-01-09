
import streamlit as st
from keras.models import load_model
from PIL import Image

from utils import classify
from utils import set_background

set_background('./background/bg.jpg')

# Set Title
st.title('Pneumonia Classification')

# Set Header
st.header('Please upload a chest X-RAY image')

# Upload File
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load Classifier
model = load_model(r"C:/Users/info/Documents/Projects/Pneumonia-classifier/model/pneumonia_classifier.h5")

# Load Class Names
with open('C:/Users/info/Documents/Projects/Pneumonia-classifier/model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Display Image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify Image
    class_name, conf_score = classify(image, model, class_names)

    # Write Classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))