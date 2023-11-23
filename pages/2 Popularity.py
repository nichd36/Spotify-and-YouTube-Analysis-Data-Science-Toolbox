import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
plt.switch_backend('Agg')
import seaborn as sns

image_path = "/Users/nichdylan/Documents/Natural Language Processing/NLP fake news/DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

def calculate_popularity_within_timeframe(df3, timeframe):
        return df3[(df3["duration"] <= timeframe) & (df3["duration"] >= timeframe - 1)]["popularity"].mean()

def popularity_duration(results_df):
        fig = go.Figure()

        scatter_trace = go.Scatter(
                x=results_df['Timeframe (minutes)'][:1],
                y=results_df['Popularity'][:1],
                mode='markers',
                name='Scatter',
                marker=dict(color='blue'),
                text=results_df['Timeframe (minutes)'][:1],
                hoverinfo='x+y+text',
        )
        fig.add_trace(scatter_trace)

        line_trace = go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Line',
                line=dict(color='red'),
                showlegend=False,
        )
        fig.add_trace(line_trace)

        fig.update_layout(
                xaxis_title='Duration (minutes)',
                yaxis_title='Average Popularity',
                legend_title='Legend',
                hovermode='closest',
        )

        frames = [go.Frame(
                data=[
                        go.Scatter(
                                x=results_df['Timeframe (minutes)'][:frame + 1],
                                y=results_df['Popularity'][:frame + 1],
                                mode='markers',
                                text=results_df['Timeframe (minutes)'][:frame + 1],
                                marker=dict(color='blue'),
                                hoverinfo='x+y+text',
                        ),
                        go.Scatter(
                                x=results_df['Timeframe (minutes)'][:frame + 1],
                                y=results_df['Popularity'][:frame + 1],
                                mode='lines',
                                line=dict(color='red'),
                                showlegend=False,
                        ),
        ],
        name=str(frame),
        ) for frame in range(1, len(results_df))]

        updatemenu = [
        {
                "buttons": [
                {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                },
                {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate",
                },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
        }
        ]

        fig.update(frames=frames)
        fig.update_layout(updatemenus=updatemenu)
        st.plotly_chart(fig, use_container_width=True)

def trend_popularity_duration(results_df):
        scatter_fig = px.scatter(
                results_df,
                x='Timeframe (minutes)',
                y='Popularity',
                labels={'Timeframe (minutes)': 'Duration (minutes)', 'Popularity': 'Average Popularity'},
                color_discrete_sequence=['blue'],
                hover_name='Timeframe (minutes)',
        )

        trendline_trace = px.scatter(
                results_df,
                x='Timeframe (minutes)',
                y='Popularity',
                trendline='ols',
                custom_data=['Timeframe (minutes)'],
                color_discrete_sequence=['red'],
        ).data[1]

        scatter_fig.add_trace(trendline_trace)
        scatter_fig.data[1].visible = 'legendonly'

        scatter_fig.update_layout(
                xaxis_title='Duration (minutes)',
                yaxis_title='Average Popularity',
                legend_title='Legend',
                hovermode='closest',
        )

        updatemenu = [
                {
                "buttons": [
                        {
                                "args": [{"visible": [True, False]}],
                                "label": "Hide Trendline",
                                "method": "update",
                        },
                        {  
                                "args": [{"visible": [True, True]}],
                                "label": "Show Trendline",
                                "method": "update",
                        },
                ],
                "direction": "down",
                "showactive": False,
                "x": 0.05,
                "xanchor": "left",
                "y": 0.05,
                "yanchor": "bottom",
                }
        ]

        scatter_fig.update_layout(
                updatemenus=updatemenu,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

def feature_importance(df3):
        numeric_columns = df3.select_dtypes(include=['number'])
        numeric_columns = numeric_columns.drop(['popularity', 'cluster', 'release_date'], axis=1)

        corr = numeric_columns.corrwith(df3["popularity"]).abs().sort_values(ascending=False)
        

        fig = px.bar(
        x=corr.values[::-1],
        y=corr.index[::-1],
        labels={'x': 'Feature Importance', 'y': 'Features'},
        orientation='h',
        color=corr.values[::-1],
        color_continuous_scale='viridis',
        )

        fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
        )

        frames = [go.Frame(
                data=[go.Bar(
                        x=corr.values[::-1][:i + 1],
                        y=corr.index[::-1][:i + 1],
                        orientation='h',
                        marker=dict(color=corr.values[::-1][:i + 1], colorscale='viridis'),
                )],
                name=str(i)
        ) for i in range(1, len(corr) + 1)]

        fig.frames = frames
        fig.update_layout(updatemenus=[
        {
        "buttons": [
        {
                "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate",
        },
        {
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": "Pause",
                "method": "animate",
        },
        ],
        "direction": "right",
        "showactive": False,
        "type": "buttons",
        "x": 1.0,
        "xanchor": "right",
        "y": -0.2,
        "yanchor": "bottom",
        }
        ])

        fig.update_layout(updatemenus=[{"buttons": [], "direction": "right", "showactive": False, "type": "buttons", "x": 1.0, "xanchor": "right", "y": -0.2, "yanchor": "bottom"}])

        st.plotly_chart(fig, use_container_width=True)

def explicitness(df3):
        df3['release_date'] = df3['release_date'].astype(str).str[-4:]

        df3['release_date'] = pd.to_datetime(df3['release_date'])
        print(df3['release_date'])

        df3['year'] = df3['release_date'].dt.year
        
        df3['release_date'] = pd.to_datetime(df3['release_date'])
        df3['year'] = df3['release_date'].dt.year.astype(str)

        df3['year'] = df3['year'].astype(int)
        print(df3['year'])

        df3['year_group'] = pd.cut(df3['year'], bins=range(1920, 2024, 5),
        labels=[f'{y}-{y+4}' for y in range(1920, 2020, 5)])

        grouped_data = df3.groupby(['year_group', 'explicit'])['popularity'].mean().unstack().reset_index()

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'bar'}]])

        not_explicit_color = 'rgba(255, 192, 203, 1)'
        explicit_color = 'rgba(255, 215, 0, 0.8)'

        fig.add_trace(go.Bar(
                x=grouped_data['year_group'],
                y=[None] * len(grouped_data),
                marker=dict(color=not_explicit_color),
                name='Not Explicit (0)',
        ))

        fig.add_trace(go.Bar(
                x=grouped_data['year_group'],
                y=[None] * len(grouped_data),
                marker=dict(color='rgba(100, 100, 100, 0.2)'),
                name='Explicit (1)',
        ))

        fig.add_trace(go.Scatter(
                x=grouped_data['year_group'],
                y=grouped_data[0],
                mode='lines+markers',
                name='Not Explicit (0)',
                line=dict(color=not_explicit_color)
        ))

        fig.add_trace(go.Scatter(
                x=grouped_data['year_group'],
                y=grouped_data[1],
                mode='lines+markers',
                name='Explicit (1)',
                line=dict(color=explicit_color)  
        ))

        fig.update_xaxes(title_text='Year Group')
        fig.update_yaxes(title_text='Average Popularity')

        frames = [go.Frame(
                data=[
                        go.Bar(
                        x=grouped_data['year_group'],
                        y=grouped_data[0] if step % 2 == 0 else grouped_data[0],
                        marker=dict(color=not_explicit_color if step % 2 == 0 else 'rgba(192, 192, 192, 1)'),
                        name='Not Explicit (0)',
                        ),
                        go.Bar(
                        x=grouped_data['year_group'],
                        y=grouped_data[1] if step % 2 == 1 else grouped_data[1],
                        marker=dict(color='rgb(254,97,97)' if step % 2 == 1 else explicit_color),
                        name='Explicit (1)',
                        ),
                ],
                name=f"Step {step}",
        ) for step in range(2)]

        animation_settings = {
                'frame': {'duration': 1000},
                'fromcurrent': True,
                'transition': {'duration': 1000},
        }

        fig.update_traces(selector=dict(name='Not Explicit (0)'), overwrite=True)
        fig.update_traces(selector=dict(name='Explicit (1)'), overwrite=True)

        fig.update(frames=frames)
        fig.update_layout(updatemenus=[
                {
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                        {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, animation_settings],
                        },
                        {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], animation_settings],
                        },
                        ],
                },
        ])
        st.plotly_chart(fig)

def main():
        df3 = load_data()

        timeFrames = []
        n = 1
        while n <= 93:
                timeFrames.append(n)
                n += 1

        st.markdown("In this section, most visualizations focus on Popularity which is our dependent variable. As aforementioned, the study aims to identify key elements influencing the success of songs over time. Hence, popularity is being a crucial factor in gauging their success across different eras.")
        calculate_popularity = [calculate_popularity_within_timeframe(df3, timeframe) for timeframe in timeFrames]
        popularity_duration_df = pd.DataFrame({'Timeframe (minutes)': timeFrames, 'Popularity': calculate_popularity})
        st.header('Popularity Based on The Duration')
        popularity_duration(popularity_duration_df)
        st.header('Average Popularity vs Duration')
        st.markdown("Below depicted the graph alongside the OLS line")
        trend_popularity_duration(popularity_duration_df)
        st.header('Feature Importance based on Popularity')
        feature_importance(df3)
        st.header('Explicitness based on Popularity on a yearly basis')
        explicitness(df3)

if __name__ == "__main__":
        main()