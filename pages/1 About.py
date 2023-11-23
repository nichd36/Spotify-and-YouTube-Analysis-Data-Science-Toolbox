import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.models import load_model

padding = 20

image_path_nicholas = "DSC_0424-Edited.jpg"
image_path_fk = "PHOTO-2023-11-23-13-00-43.jpg"
image_path_josh = "PHOTO-2023-11-23-14-59-50.jpg"

st.markdown(
    """
    <style>
   .sidebar .sidebar-content {
        background: url({image_path})
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Our team')

if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

col1, col2, col3 = st.columns([3,3,3])

with col1:
        st.image(image_path_nicholas, caption="Nicholas Dylan 0133646", use_column_width=True)
with col2:
        st.image(image_path_fk, caption="Foo Fang Khai 0134196", use_column_width=True)
with col3:
        st.image(image_path_josh, caption="Joshua Tham 0133885", use_column_width=True)


st.header('Abstract')

st.markdown("This research investigates the evolution of musical genres from the 1920s to modern pop, aiming to uncover the relationship between cultural shifts and musical success. Focusing on elements like energy, tempo, rhythm, and lyrics, it explores the enduring appeal of earlier works in today's music landscape.")
st.markdown("Through Machine Learning analysis, it seeks to identify key components influencing musical success, offering valuable insights for artists and producers.")
            