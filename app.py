from dash import Dash, html
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import datetime
import numpy as np


model_danceability = load_model('/Users/nichdylan/Downloads/DashDataScienceToolbox/model_danceability.h5')

app = Dash(__name__)

df3 = pd.read_csv('/Users/nichdylan/Downloads/DashDataScienceToolbox/df3.csv')

year_options_start = [{"label": str(year), "value": year} for year in range(1922, 2041)]
year_options_end = [{"label": str(year), "value": year} for year in range(1923, 2042)]

menu_dropdown = dcc.Dropdown(
    id="menu-dropdown-component",
    options=[
        {"label": "Prediction", "value": "Prediction"},
        {"label": "Current data", "value": "Current data"},
    ],
    clearable=False,
    value="Prediction",
)

menu_dropdown_text = html.P(
    id="menu-dropdown-text", children=["Menu", html.Br(), " Dashboard"]
)

menu_title = html.H1(id="menu-name", children="")

menu_body = html.P(
    className="menu-description", id="menu-description", children=[""]
)

side_panel_layout = html.Div(
    id="panel-side",
    style={
        'float': 'left',  # Float the div to the left
        'width': '100%',
        'padding': '20px',
        'background': '#aac9fa',
        'box-shadow': '20px 20px 60px #b6b6b6, -20px -20px 60px #ffffff',
        'height': '100vh',  # Adjust the height as needed
        'overflow': 'auto',  # Add scrolling if content overflows
    },
    children=[
        menu_dropdown_text,
        html.Div(id="menu_dropdown", children=menu_dropdown),
        html.Div(id="panel-side-text", children=[menu_title, menu_body]),
    ],
)

main_panel_layout = html.Div([
    html.Div([
        html.H1("Danceability prediction for the next 15 years", style={'text-align':'center', 'font-family': 'Helvetica'}),
        dcc.Graph(id='danceability-chart',
                  style={
                      'width': '80%',
                      'margin': '0 auto',
                  }),

    html.Div([
        html.Div([
            html.Label("Start Year", style={'text-align': 'center', 'font-family': 'Helvetica', 'margin-right': '10px'}),
            dcc.Dropdown(id='start-year',
                         value=1922,
                         options=year_options_start,
                         style={'font-family': 'Helvetica', 'padding-right': '10px', ' margin-left': '10px'}
                        ),
        ], style={'display': 'inline-block', 'width': '100%', 'margin-right': '10px', 'margin': '0 auto'}),

        html.Div([
            html.Label("End Year", style={'text-align': 'center', 'font-family': 'Helvetica', 'margin-right': '10px'}),
            dcc.Dropdown(id='end-year',
                         value=2040,
                         options=year_options_end,
                         style={'font-family': 'Helvetica', 'width': '100%', 'padding-left': '10px', 'margin-right': '10px'}
                        ),
        ], style={'display': 'inline-block', 'margin-left': '10px', 'margin': '0 auto', 'width': '100%'}),
    ], style={'display': 'flex', 'align-items': 'center', 'margin-top': '20px', 'margin': '0 auto', 'margin-left': '10px', 'margin-right': '10px'}),

    html.Div(id='error-message', style={'color': 'red',  'font-family': 'Helvetica'}),

    ], style={'border-radius': '50px', 'background': '#f0f0f0', 'box-shadow': '20px 20px 60px #b6b6b6, -20px -20px 60px #ffffff', 'margin': '0 auto', 'justify-content': 'center'})
]
, style={
    'justify-content': 'center', 
    'width': '90%',
    'float': 'right',
    'align-items': 'center', 
    'background': '#f0f0f0', 
    'height': '100vh',
    'margin': '0'}
)

app.layout = html.Div([
    html.Div([
        side_panel_layout,
    ], style={'width': '20%', 'display': 'inline-block'}),  # Use display:inline-block for side panel
    
    html.Div([
        main_panel_layout,
    ], style={'width': '80%', 'display': 'inline-block'}),  # Use display:inline-block for main panel
])

@app.callback(
    Output('start-year', 'value'),
    Output('end-year', 'value'),
    Output('error-message', 'children'),
    Input('start-year', 'value'),
    Input('end-year', 'value')
)

def update_years(start_year, end_year):
    error_message = ""

    if start_year is None or end_year is None:
        return start_year, end_year, error_message

    if start_year >= end_year:
        end_year = start_year + 1
        error_message = "Error: Start year must be less than end year."

    return start_year, end_year, error_message


@app.callback(
    Output('danceability-chart', 'figure'),
    [Input('start-year', 'value'),
     Input('end-year', 'value')]
)

def update_chart(start_year, end_year):
    if start_year is None or end_year is None or start_year >= end_year:
        return {}

    predict_data = df3[['danceability', 'release_date']]
    predict_data = predict_data.groupby('release_date')['danceability'].mean().reset_index()
    predict_data = predict_data.drop(predict_data.index[:1])

    danceability_data = predict_data['danceability'].values
    danceability_data = danceability_data.reshape((-1,1))

    look_back = 3

    num_future_steps = 15

    date_danceability = predict_data['release_date']
    danceability_data = danceability_data.flatten()

    forecasted_danceability = []

    input_sequence = np.array(danceability_data[-look_back:]).reshape((1, look_back, 1))

    start_year_predict = 2022

    for i in range(num_future_steps):
        predicted_value = model_danceability.predict(input_sequence)[0, 0]
        
        forecasted_danceability.append(predicted_value)
        
        input_sequence = np.squeeze(input_sequence)
        input_sequence = np.append(input_sequence[1:], predicted_value)
        input_sequence = input_sequence.reshape((1, look_back, 1))

    forecasted_danceability = np.array(forecasted_danceability)

    forecasted_years = [start_year_predict + i for i in range(num_future_steps)]

    trace1 = go.Scatter(
        x = date_danceability,
        y = danceability_data,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = forecasted_years,
        y = forecasted_danceability,
        mode = 'lines',
        name = 'Prediction'
    )
    layout = go.Layout(
        xaxis = {'title' : "Year", 'range': [start_year, end_year]},
        yaxis = {'title' : "Close", 'range': [0, 1]},
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

if __name__ == '__main__':
    app.run(debug=True)