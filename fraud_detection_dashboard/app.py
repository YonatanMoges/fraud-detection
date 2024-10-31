from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize Flask app and Dash app
server = Flask(__name__)

try:
    # Attempt to load data
    data = pd.read_csv('../data/processed/processed_fraud_data.csv')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    data = pd.DataFrame()  # Initialize empty DataFrame if loading fails

app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Ensure data is not empty and contains required columns
if not data.empty and all(col in data.columns for col in ['class', 'device_id', 'browser', 'country_x']):
    # Calculate summary statistics
    total_transactions = len(data)
    fraud_cases = data['class'].sum()  # Assuming 'class' = 1 indicates fraud
    fraud_percentage = (fraud_cases / total_transactions) * 100
else:
    total_transactions, fraud_cases, fraud_percentage = 0, 0, 0

# Dash layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard"),
    
    # Summary statistics
    html.Div(children=[
        html.H3(f"Total Transactions: {total_transactions}"),
        html.H3(f"Total Fraud Cases: {fraud_cases}"),
        html.H3(f"Fraud Percentage: {fraud_percentage:.2f}%")
    ], style={'textAlign': 'center'}),

    # Fraud by Device and Browser bar chart
    dcc.Graph(id='device-browser-chart'),

    # Fraud by Country (Geographic Distribution)
    dcc.Graph(id='geo-chart')
])

# Callback to update device-browser bar chart
@app.callback(
    Output('device-browser-chart', 'figure'),
    Input('device-browser-chart', 'id')
)
def update_device_browser_chart(_):
    if data.empty:
        return px.bar(title="No Data Available")
    try:
        device_data = data[data['class'] == 1]
        device_browser_counts = device_data.groupby(['device_id', 'browser']).size().reset_index(name='count')
        fig = px.bar(device_browser_counts, x='device_id', y='count', color='browser',
                     labels={'device_id': 'Device', 'count': 'Number of Fraud Cases'},
                     title="Fraud Cases by Device and Browser")
        return fig
    except Exception as e:
        print(f"Error generating device-browser chart: {e}")
        return px.bar(title="Error in Generating Chart")

# Callback to update geographic distribution of fraud
@app.callback(
    Output('geo-chart', 'figure'),
    Input('geo-chart', 'id')
)
def update_geo_chart(_):
    if data.empty:
        return px.choropleth(title="No Data Available")
    try:
        geo_data = data[data['class'] == 1]
        geo_counts = geo_data['country_x'].value_counts().reset_index()
        geo_counts.columns = ['country', 'fraud_cases']
        fig = px.choropleth(geo_counts, locations='country', locationmode='country names', color='fraud_cases',
                            title="Geographic Distribution of Fraud Cases",
                            labels={'fraud_cases': 'Number of Fraud Cases'})
        return fig
    except Exception as e:
        print(f"Error generating geographic chart: {e}")
        return px.choropleth(title="Error in Generating Chart")

# Run server
if __name__ == '__main__':
    server.run(debug=True)
