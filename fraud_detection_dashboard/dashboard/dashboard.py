import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

def init_dashboard(server):
    app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')
    
    # Fetch summary data from Flask API
    summary_data = requests.get("http://localhost:5000/api/summary").json()
    total_transactions = summary_data["total_transactions"]
    fraud_cases = summary_data["fraud_cases"]
    fraud_percentage = summary_data["fraud_percentage"]

    app.layout = html.Div(style={"fontFamily": "Arial", "padding": "20px"}, children=[
        html.H1("Fraud Detection Dashboard", style={"textAlign": "center", "color": "#2c3e50"}),

        html.Div([
            html.Div(f"Total Transactions: {total_transactions}", className="summary-box"),
            html.Div(f"Fraud Cases: {fraud_cases}", className="summary-box"),
            html.Div(f"Fraud Percentage: {fraud_percentage:.2f}%", className="summary-box"),
        ], className="summary-section", style={"display": "flex", "justifyContent": "space-around", "padding": "10px", "backgroundColor": "#ecf0f1"}),

        dcc.Graph(id='fraud-trend-chart'),
        dcc.Graph(id='fraud-geography-chart'),
        dcc.Graph(id='device-browser-chart'),
    ])

    @app.callback(Output('fraud-trend-chart', 'figure'), Input('fraud-trend-chart', 'id'))
    def update_fraud_trend_chart(_):
        trend_data = requests.get("http://localhost:5000/api/fraud_trends").json()
        trend_df = pd.DataFrame(list(trend_data.items()), columns=['Date', 'Fraud Cases'])
        trend_df['Date'] = pd.to_datetime(trend_df['Date'])
        fig = px.line(trend_df, x='Date', y='Fraud Cases', title="Fraud Cases Over Time")
        fig.update_layout(plot_bgcolor="#f0f0f0", paper_bgcolor="#ffffff")
        return fig

    @app.callback(Output('fraud-geography-chart', 'figure'), Input('fraud-geography-chart', 'id'))
    def update_fraud_geography_chart(_):
        geo_data = requests.get("http://localhost:5000/api/fraud_geography").json()
        geo_df = pd.DataFrame(list(geo_data.items()), columns=['Location', 'Fraud Cases'])
        fig = px.bar(geo_df, x='Location', y='Fraud Cases', title="Fraud Cases by Location")
        fig.update_layout(plot_bgcolor="#f0f0f0", paper_bgcolor="#ffffff")
        return fig

    @app.callback(Output('device-browser-chart', 'figure'), Input('device-browser-chart', 'id'))
    def update_device_browser_chart(_):
        # Generate or read example data for device-browser comparisons
        device_browser_data = fraud_data[fraud_data['is_fraud'] == 1].groupby(['device', 'browser']).size().reset_index(name='Fraud Cases')
        fig = px.bar(device_browser_data, x='device', y='Fraud Cases', color='browser', title="Fraud Cases by Device and Browser")
        fig.update_layout(plot_bgcolor="#f0f0f0", paper_bgcolor="#ffffff")
        return fig

    return app
