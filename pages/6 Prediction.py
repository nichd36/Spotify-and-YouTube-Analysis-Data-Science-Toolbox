import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model

image_path = "/Users/nichdylan/Documents/Natural Language Processing/NLP fake news/DSC_0424-Edited.jpg"
st.set_page_config(layout="wide", page_title="Data Science Toolbox", page_icon = image_path)

model_danceability = load_model('model_danceability.h5')
model_acousticness = load_model('model_acousticness.h5')
model_energy = load_model('model_energy.h5')

st.title('Prediction ⚡️🎼💃')

def load_data():
        df3 = pd.read_csv("df3.csv")
        df3 = df3[df3['release_date'] != 1900]
        return df3

def prediction(feature, start_year, end_year):
        df3 = load_data()

        predict_data = df3[[feature, 'release_date']]
        predict_data = predict_data.groupby('release_date')[feature].mean().reset_index()

        feature_data = predict_data[feature].values
        feature_data = feature_data.reshape((-1, 1))

        if feature == "energy":
                look_back = 10
        elif feature == "acousticness":
                look_back = 15
        else:
                look_back = 3

        num_future_steps = 15

        date_feature = predict_data['release_date']
        feature_data = feature_data.flatten()

        forecasted_feature = []

        input_sequence = np.array(feature_data[-look_back:]).reshape((1, look_back, 1))

        start_year_predict = 2022
        model_name = "model_"+ feature
        model = eval(model_name)

        for i in range(num_future_steps):
                predicted_value = model.predict(input_sequence)[0, 0]

                forecasted_feature.append(predicted_value)

                input_sequence = np.squeeze(input_sequence)
                input_sequence = np.append(input_sequence[1:], predicted_value)
                input_sequence = input_sequence.reshape((1, look_back, 1))

        forecasted_feature = np.array(forecasted_feature)

        if feature != "danceability":
                forecasted_feature = forecasted_feature.reshape((-1))

        forecasted_years = [start_year_predict + i for i in range(num_future_steps)]

        feature_data_df = pd.DataFrame({feature: feature_data, 'Year': date_feature})
        feature_forecast_df = pd.DataFrame({'Forecast': forecasted_feature, 'Year': forecasted_years})

        all_years = df3['release_date'].unique()
        all_years = np.arange(min(all_years), max(all_years) + 16)

        feature_data_df = pd.DataFrame({'Year': all_years}).merge(feature_data_df, on='Year', how='left')
        feature_forecast_df = pd.DataFrame({'Year': all_years}).merge(feature_forecast_df, on='Year', how='left')

        chart_data = feature_data_df.merge(feature_forecast_df, on='Year', how='outer')
        chart_data = chart_data[(chart_data['Year'] >= start_year) & (chart_data['Year'] <= end_year)]
     
        st.subheader("Data on the annual average " + feature + " value and predictions")
        st.line_chart(chart_data.set_index('Year')[[feature, 'Forecast']], use_container_width=True)

def main():
        df3 = load_data()

        # st.dataframe(df3)

        st.sidebar.header('Year Selection')

        min_year = 1925
        max_year = df3['release_date'].max()

        start_year = st.sidebar.number_input('Start Year', min_value=min_year, max_value=max_year+14)
        end_year = st.sidebar.number_input('End Year', min_value=min_year+1, max_value=df3['release_date'].max()+15, value=2036)

        if start_year >= end_year:
                st.sidebar.error("Error: Start year must be less than end year.")
                return

        prediction("danceability", start_year, end_year)
        prediction("acousticness", start_year, end_year)
        prediction("energy", start_year, end_year)
        
if __name__ == "__main__":
        main()