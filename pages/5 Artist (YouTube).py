import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from tensorflow import keras
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
plt.switch_backend('Agg')
import seaborn as sns

image_path = "DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

st.title('YouTube')

def top_ten_artists(df3):
        df3['Views'] = df3['Views'].astype(int) 
        df_top_10_artists = df3.drop_duplicates(subset='Artist').nlargest(10, 'Views')

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        fig.add_trace(go.Scatterpolar(r=df_top_10_artists['Views'], theta=df_top_10_artists['Artist'], fill='toself'), row=1, col=1)
        fig.update_traces(fillcolor = 'lightblue')

        fig.update_layout(height=600, width=800, title_text="Top 10 Artists vs Views")
        st.plotly_chart(fig)

def ytb_data(ytb_data):
        columns_to_drop = ["Unnamed: 0", "Url_spotify", "Album_type", "Uri", "Danceability", "Energy", "Key", "Loudness", "Album", "Title", "Channel", "Likes", "Comments", "Description", "Licensed", "official_video", "Stream", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Url_youtube"]
        ytb_data = ytb_data.drop(columns = columns_to_drop)
        
        ytb_data["Views"].fillna(ytb_data["Views"].mean(), inplace = True)
        ytb_data.dropna()
        
        ytb_data["Duration_ms"] = ytb_data["Duration_ms"] / 60000
        ytb_data.rename(columns = {"Duration_ms" : "Duration"}, inplace = True)

        ytb_data["Views"] = pd.to_numeric(ytb_data["Views"], errors = "coerce")

        # st.write(ytb_data.isnull().sum())
        print("\n\n")

        return ytb_data

def calculate_views_within_timeframe(youtube, timeframe):
        return youtube[(youtube["Duration"] <= timeframe) & (youtube["Duration"] >= timeframe - 0.25)]["Views"].mean()

def views_youtube(youtube, results_df, timeFrames):
        calculate_views = [calculate_views_within_timeframe(youtube, timeframe) for timeframe in timeFrames]
        results_df = pd.DataFrame({'Timeframe (minutes)': timeFrames, 'Views': calculate_views})

        fig = px.scatter(
                results_df,
                x = 'Timeframe (minutes)',
                y = 'Views',
                labels = {'Timeframe (minutes)': 'Duration (minutes)', 'Views': 'Average Views'},
                color_discrete_sequence = ['blue'], 
                hover_name = 'Timeframe (minutes)',
        )

        fig.add_trace(
        px.scatter(
                results_df,
                x = 'Timeframe (minutes)',
                y = 'Views',
                trendline = 'ols',
                custom_data = ['Timeframe (minutes)'],
                color_discrete_sequence = ['red'], 
        ).data[1]
        )

        fig.update_layout(
                xaxis_title = 'Duration (minutes)',
                yaxis_title = 'Average Views',
                legend_title = 'Legend',
                hovermode = 'closest',
        )

        st.plotly_chart(fig)

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

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
        youtube = pd.read_csv("Spotify_Youtube.csv")
        youtube = ytb_data(youtube)
        print(youtube.isnull().sum())

        st.dataframe(youtube)
        panjang = youtube.shape[0]
        st.write('After data cleaning, the YouTube dataframe consists of', panjang, 'rows.')

        views_data_type = youtube['Views'].dtype

        print("\n\n\nData type of the 'Views' column:", views_data_type)
        
        top_ten_artists(youtube)

        timeFrames = []

        n = 0.25
        while n <= 78:
                timeFrames.append(n)
                n += 0.25

        calculate_views = [calculate_views_within_timeframe(youtube, timeframe) for timeframe in timeFrames]
        results_df = pd.DataFrame({'Timeframe (minutes)': timeFrames, 'Views': calculate_views})

        options = ["Luis Fonsi", "Wiz Khalifa", "Katy Perry"]
        initial_value = "Luis Fonsi"
        selected_option = st.selectbox('Select an artist:', options, index=options.index(initial_value))

        characteristics(selected_option)

        st.header("Views based on the duration")
        st.markdown("Below depicted the graph alongside the OLS line")
        views_youtube(youtube, results_df, timeFrames)

if __name__ == "__main__":
        main()
