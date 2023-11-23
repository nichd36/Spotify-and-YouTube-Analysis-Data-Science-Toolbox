import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
plt.switch_backend('Agg')
import seaborn as sns

image_path = "/Users/nichdylan/Documents/Natural Language Processing/NLP fake news/DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

model_danceability = load_model('model_danceability.h5')
model_acousticness = load_model('model_acousticness.h5')
model_energy = load_model('model_energy.h5')

st.title('Data Science Toolbox')

artist = pd.read_csv("artists.csv")
tracks = pd.read_csv("tracks.csv")
youtube = pd.read_csv("Spotify_Youtube.csv")

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

def main():
        df3 = load_data()

        st.markdown("Our spotify dataset was made from a combination of 2 dataset, one for artist, and the other one for tracks.")
        st.header("Spotify's artists dataset 👩‍🎤👨🏻‍🎤🧑🏾‍🎤🎤")
        st.markdown('to download our artist dataset, visit: www.google.com')
        st.dataframe(artist)
        panjang = artist.shape[0]
        st.write('This dataframe consists of', panjang, 'rows.')

        st.header("Spotify's tracks dataset 🎵🪗")
        st.markdown('to download our tracks dataset, visit: www.google.com')
        st.dataframe(tracks)
        panjang = tracks.shape[0]
        st.write('This dataframe consists of', panjang, 'rows.')

        st.header('A glimpse of our final data 📊')
        st.markdown('to download our dataset, visit: www.google.com')
        st.dataframe(df3)
        panjang = df3.shape[0]
        st.write('This dataframe consists of', panjang, 'rows.')
        st.markdown("The number of rows was reduced through actions such as removing duplicate data and eliminating null values.")

        st.header('Heatmap to show correlations')
        st.markdown("The correlation analysis showed that loudness and energy are highly correlated, with danceability and valence following closely. Based on insights from a study by Millecamp et al. (2018), energy was chosen over loudness for prediction, considering participants' preference. Additionally, the Feature Importance chart which will be show later favors danceability over valence. Thus, for the next 15 years, predictions will focus on acousticness, energy, and danceability to guide music creators in crafting more popular songs.")
        numeric_df = df3.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        plt.figure(figsize=(16, 9))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)

        st.header('Other: Youtube 💽')
        st.markdown('to download our dataset, visit: www.google.com')
        st.dataframe(youtube)
        panjang = youtube.shape[0]
        st.write('This dataframe consists of', panjang, 'rows.')
        
if __name__ == "__main__":
        main()