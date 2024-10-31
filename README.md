# Fraud Detection Project

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing](#1-preprocessing)
  - [Model Training](#2-model-training)
  - [Model Explainability](#3-model-explainability)
  - [API Deployment](#4-api-deployment)
  - [Dashboard Development](#5-dashboard-development)
- [Conclusion](#conclusion)


## Overview
This project, developed for Adey Innovations Inc., aims to enhance fraud detection for e-commerce and bank credit transactions using machine learning techniques. The objective is to accurately identify fraudulent transactions by analyzing transaction data, patterns, and geolocation information. This project includes data preprocessing, model training, model explainability, API deployment, and dashboard development.

## Project Structure

``` 

fraud-detection/
├── data/
│   ├── Fraud_Data.csv                 # Raw e-commerce transaction data
│   ├── IpAddress_to_Country.csv       # IP address to country mapping data
│   ├── creditcard.csv                 # Raw bank transaction data
│
├── fraud_detection_api/
│   ├── models/                        # Directory for model storage in API deployment
│   ├── Dockerfile                     # Dockerfile for API containerization
│   ├── requirements.txt               # Dependencies for API environment
│   └── serve_model.py                 # Flask API script for serving fraud detection model
│
├── fraud_detection_dashboard/
│   ├── dashboard/                     # Dashboard components and main dashboard script
│   ├── static/                        # Static assets for dashboard (CSS, JS)
│   ├── templates/                     # HTML templates for dashboard UI
│   ├── app.py                         # Main application script for the dashboard
│   └── requirements.txt               # Dependencies for dashboard environment
│
├── models/                            # Directory for storing trained models
├── myenv/                             # Virtual environment folder (if applicable)
├── notebooks/
│   ├── preprocessing.ipynb            # Notebook for demonstrating preprocessing and EDA
│   ├── model_training.ipynb           # Notebook for demonstrating model training and evaluation
│   └── model_explainability.ipynb     # Notebook for model explainability (SHAP and LIME)
│
├── scripts/
│   ├── preprocessing.py               # Script for data preprocessing and feature engineering
│   ├── model_training.py              # Script for model training and evaluation
│   └── model_explainability.py        # Script for SHAP and LIME model explainability
│
├── tests/                             # Directory for test scripts
├── src/                               # Source files and utility functions
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
└── requirements.txt                   # Project dependencies
```
## Datasets
1. **Fraud_Data.csv** - Contains details about e-commerce transactions, including transaction amount, device, source, browser, and IP address.  
2. **IpAddress_to_Country.csv** - Maps IP addresses to countries, used for geolocation analysis.  
3. **creditcard.csv** - Contains anonymized bank transaction data with PCA-transformed features.
## Features
**Data Preprocessing**: Handles missing values, removes duplicates, performs data cleaning, and merges datasets for geolocation analysis.  
**Feature Engineering**: Includes creating features like transaction frequency, velocity, time-based features (hour and day), and encoding categorical variables.  
**Modeling and Evaluation**: Builds and evaluates machine learning models on e-commerce and bank credit transaction data.  
**Model Explainability**: Uses SHAP and LIME to explain model predictions and feature importance.  
**API Deployment**: Serves the fraud detection model via a Flask API using Docker.  
**Dashboard Development**: A dashboard built with Flask and Dash to visualize fraud trends and model insights.  

## Installation
1. Clone the repository:

```bash
git clone https://github.com/YonatanMoges/fraud-detection.git
cd fraud-detection
```
2. Install required dependencies:

```bash
pip install -r requirements.txt
```
## Usage

1. **Preprocessing**: The `scripts/preprocessing.py` file contains a FraudPreprocessing class for preprocessing both e-commerce and credit card transaction data. The `notebooks/preprocessing.ipynb` notebook provides an example of running preprocessing and performing exploratory data analysis.

2. **Model Training**: The `scripts/model_training.py` script includes the FraudDetectionModels class, which handles training and evaluating various machine learning models. Run this script to train models on the preprocessed data. For detailed results and insights, refer to `notebooks/model_training.ipynb`.

3. **Model Explainability**: The `scripts/model_explainability.py` script uses SHAP and LIME to explain model predictions and feature importance. The `notebooks/model_explainability.ipynb` notebook demonstrates generating interpretability plots for the models.

4. **API Deployment**: The fraud_detection_api directory contains files for deploying the fraud detection model as a Flask API. To build and run the API using Docker:

```bash
# Navigate to the fraud_detection_api directory
cd fraud_detection_api
```
```
# Build the Docker image
docker build -t fraud-detection-api .
```
```
# Run the Docker container
docker run -p 5000:5000 fraud-detection-api
```
Once running, you can make predictions by sending a POST request to the API:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "features": [
    0.05514027570137852, 0.17241379310344832, 0.6466963865028451, 1.0, 0.0, 1.0, 
    0.3620689655172414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.43467338016281104, 
    0.08695652173913043, 0.8333333333333333, 0.0, 0.0
  ]
}'
```
5. **Dashboard Development**: The fraud_detection_dashboard directory contains the necessary files to launch a Flask and Dash-based dashboard for visualizing fraud detection insights.

To run the dashboard:

``` bash
# Navigate to the fraud_detection_dashboard directory
cd fraud_detection_dashboard
```
```
# Install dependencies
pip install -r requirements.txt
```
```
# Start the dashboard
python app.py
```
Conclusion
This project provides an end-to-end solution for fraud detection on e-commerce and bank transactions. It includes data preprocessing, model training, explainability, deployment through a Flask API, and an interactive dashboard to gain insights into fraud trends and model behavior.