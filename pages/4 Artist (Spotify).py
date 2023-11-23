import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
plt.switch_backend('Agg')
import seaborn as sns

image_path = "DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

st.title('Spotify')

artist = pd.read_csv("artists.csv")

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

def top_ten_artists(df3):
        df3['followers'] = df3['followers'].astype(int) 
        df_top_10_artists = df3.drop_duplicates(subset='artists').nlargest(10, 'followers')

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        fig.add_trace(go.Scatterpolar(r=df_top_10_artists['followers'], theta=df_top_10_artists['artists'], fill='toself'), row=1, col=1)
        fig.update_traces(fillcolor = 'lightblue')

        fig.update_layout(height=600, width=800, title_text="Top 10 Artists vs Followers")
        st.plotly_chart(fig)

def characteristics(artist):
        df3 = load_data()

        selected_songs = df3[df3['artists'] == artist]
        numerical_columns = selected_songs.select_dtypes(include=[int, float])

        print(numerical_columns)

        grouped_data = numerical_columns.groupby('release_date').mean()

        fig = px.line(grouped_data, x=grouped_data.index, y=['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'loudness'],
                  labels={'release_date': 'Release Date', 'value': 'Characteristic Value'},
                  title=f"{artist} Song Characteristics Over Time")
        
        st.plotly_chart(fig)

def main():
        top_ten_artists(load_data())
        st.header('👇👩‍🎤👨🏻‍🎤🧑🏾‍🎤🎤')

        options = ["Ed Sheeran", "Ariana Grande", "Drake", "Justin Bieber", "Eminem"]
        initial_value = "Ed Sheeran"
        selected_option = st.selectbox('Select an artist:', options, index=options.index(initial_value))

        characteristics(selected_option)

        
if __name__ == "__main__":
        main()
