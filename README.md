# Fraud Detection Project

## Overview

This project, developed for **Adey Innovations Inc.**, aims to enhance fraud detection for e-commerce transactions and bank credit transactions using machine learning techniques. The main objective is to build and deploy models that accurately identify fraudulent transactions by analyzing transaction data, patterns, and geolocation information.

## Project Structure
``` bash
fraud-detection/
├── data/
│   ├── Fraud_Data.csv                # Raw e-commerce transaction data
│   ├── IpAddress_to_Country.csv      # IP address to country mapping data
│   ├── creditcard.csv                # Raw bank transaction data
│
├── notebooks/
│   └── preprocessing.ipynb                    # Notebook for demonstrating preprocessing and analysis
│
├── scripts/
│   └── preprocessing.py              # Script for data preprocessing and feature engineering
│
├── README.md                         # Project documentation
├── requirements.txt                  # Project dependencies for installation
└── .gitignore                        # Files and directories to ignore in Git
```

## Datasets

1. **Fraud_Data.csv** - Contains details about e-commerce transactions, including transaction amount, device, source, browser, and IP address.
2. **IpAddress_to_Country.csv** - Maps IP addresses to countries, used for geolocation analysis.
3. **creditcard.csv** - Contains anonymized bank transaction data with PCA-transformed features.

## Features

- **Data Preprocessing**: Handles missing values, removes duplicates, performs data cleaning, and merges datasets for geolocation analysis.
- **Feature Engineering**: Includes creating features like transaction frequency, velocity, time-based features (hour and day), and encoding categorical variables.
- **Modeling and Evaluation**: Building and evaluating machine learning models for real-time fraud detection.

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

1. **Preprocessing**:
   The `scripts/preprocessing.py` file contains a `FraudPreprocessing` class, which includes methods to preprocess both e-commerce and credit card transaction data.

   ```python
   from scripts.preprocessing import FraudPreprocessing
    ```
   # Initialize preprocessor
   ```python
   preprocessor = FraudPreprocessing()
    ```
   # Run all preprocessing steps
   ```python
   fraud_data_processed, credit_data_processed = preprocessor.preprocess_all()
    ```
   # Save processed datasets
   ```python
   preprocessor.save_processed_data("processed_fraud_data.csv", "processed_credit_data.csv")
    ```
    
2. **Demo**: 
    The `notebooks/preprocessing.ipynb` notebook provides an example of running preprocessing, and performing exploratory data analysis.