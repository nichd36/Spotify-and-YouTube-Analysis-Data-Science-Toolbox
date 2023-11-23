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

image_path = "/Users/nichdylan/Documents/Natural Language Processing/NLP fake news/DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

st.title('Song clusters vs Popularity')

def popularity_cluster(df3):
        st.header('Clustering the songs based on popularity')

        cluster_stats = df3.groupby('cluster').agg({'popularity': 'mean', 'name': 'count'}).reset_index()
        cluster_stats.rename(columns = {'popularity': 'Mean Popularity', 'name': 'Number of Songs'}, inplace = True)

        fig = px.scatter(
        cluster_stats,
        x = 'cluster',
        y = 'Mean Popularity',
        size = 'Number of Songs',  
        color = 'Number of Songs', 
        labels = {'cluster': 'Cluster', 'Mean Popularity': 'Mean Popularity', 'Number of Songs': 'Number of Songs'},
        hover_name = 'cluster',
        size_max = 40,
        color_continuous_scale = 'Rainbow'
        )

        fig.update_xaxes(title_text = 'Cluster')
        fig.update_yaxes(title_text = 'Mean Popularity')
        st.plotly_chart(fig, use_container_width=True)

def popular_songs(df3):
        st.header('A glimpse into the most popular songs')
        cluster_8_songs = df3[df3['cluster'] == 8]

        for song in cluster_8_songs['name'][:8]:
                st.caption(f"- {song}")
        
        average_duration_cluster_8 = cluster_8_songs['duration'].mean()
        st.write("Average Duration for Cluster 8 Songs: " + str(average_duration_cluster_8) + " minutes")

def k_means(df3):
        st.header("Clustered using K-Means without elbow method")
        warnings.filterwarnings("ignore")
        X = df3[['duration', 'popularity']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        K = 15 

        kmeans = KMeans(n_clusters=K, random_state=42)
        df3['cluster'] = kmeans.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        for i, center in enumerate(cluster_centers):
                st.write(f"Cluster {i+1} Center - Duration: {center[0]:.2f} minutes, Popularity: {center[1]:.2f}")

        warnings.filterwarnings("ignore")
        ssd = []
        K_range = range(1, 20)

        for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                ssd.append(kmeans.inertia_)

        plt.plot(K_range, ssd, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Distances (SSD)')
        st.header('Implementing the Elbow Method for Optimal K')
        # st.plotly_chart(plt)
        elbow_data = pd.DataFrame({'K': K_range, 'SSD': ssd})

        # Create the elbow plot using Plotly
        fig = px.line(elbow_data, x='K', y='SSD')
        st.plotly_chart(fig, use_container_width=True)

def threed_k_means(df3):
        warnings.filterwarnings("ignore")
        X = df3[['duration', 'popularity']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        K = 8 

        kmeans = KMeans(n_clusters=K, random_state=42)
        df3['cluster'] = kmeans.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        for i, center in enumerate(cluster_centers):
                st.write(f"Cluster {i+1} Center - Duration: {center[0]:.2f} minutes, Popularity: {center[1]:.2f}")
        
        clustered_data = df3.copy()
        fig = px.scatter_3d(clustered_data, x='duration', y='popularity', z='cluster',
                        color='cluster', opacity=0.7, size_max=10,
                        title='K-Means Clustering of Songs',
                        labels={'duration': 'Duration', 'popularity': 'Popularity', 'cluster': 'Cluster'},
                        hover_data=['cluster'])

        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(zaxis=dict(range=[0, K])))
        st.plotly_chart(fig)
        st.markdown("A three-dimensional graph created with KMeans to offer various insights, including observations on cluster patterns, the degree of separation between clusters, and more. It's evident that cluster 4 stands out with the most tightly grouped points, signifying that the songs in this cluster exhibit more uniform characteristics compared to other clusters. In contrast, the other clusters show more separated points, suggesting a greater diversity within those clusters.")

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

def main():
        df3 = load_data()

        popularity_cluster(df3)
        popular_songs(df3)
        k_means(df3)
        threed_k_means(df3)

if __name__ == "__main__":
        main()