# scripts/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class FraudPreprocessing:
    def __init__(self, fraud_data_path, ip_data_path, credit_data_path):
        # Load datasets
        self.fraud_data = pd.read_csv(fraud_data_path)
        self.ip_data = pd.read_csv(ip_data_path)
        self.credit_data = pd.read_csv(credit_data_path)
        
    def handle_missing_values(self):
        # Impute or drop missing values for each dataset
        self.fraud_data.dropna(inplace=True)
        self.ip_data.dropna(inplace=True)
        self.credit_data.dropna(inplace=True)
    
    def data_cleaning(self):
        # Remove duplicates and correct data types
        self.fraud_data.drop_duplicates(inplace=True)
        self.credit_data.drop_duplicates(inplace=True)
        
        # Convert datetime fields
        self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'])
        self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
    
    def univariate_analysis(self):
        # Univariate analysis for basic insights
        fraud_counts = self.fraud_data['class'].value_counts(normalize=True)
        credit_fraud_counts = self.credit_data['Class'].value_counts(normalize=True)
        return {"Fraud Counts E-commerce": fraud_counts, "Fraud Counts Credit": credit_fraud_counts}
    
    def bivariate_analysis(self):
        # Bivariate analysis - Correlation and some simple comparisons
        # Select only numeric columns for correlation matrix calculation
        fraud_numeric = self.fraud_data.select_dtypes(include=[np.number])
        credit_numeric = self.credit_data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrices
        corr_matrix_fraud = fraud_numeric.corr()
        corr_matrix_credit = credit_numeric.corr()
        
        return {"Fraud Data Correlation": corr_matrix_fraud, "Credit Data Correlation": corr_matrix_credit}

    def merge_datasets(self):
        # Ensure IP addresses are strings and handle missing values
        self.fraud_data['ip_address'] = self.fraud_data['ip_address'].fillna("0.0.0.0").astype(str)
        self.ip_data['lower_bound_ip_address'] = self.ip_data['lower_bound_ip_address'].fillna("0.0.0.0").astype(str)
        self.ip_data['upper_bound_ip_address'] = self.ip_data['upper_bound_ip_address'].fillna("0.0.0.0").astype(str)
        
        # Convert IP addresses to integer format
        self.fraud_data['ip_address'] = self.fraud_data['ip_address'].apply(self.ip_to_integer)
        self.ip_data['lower_bound_ip_address'] = self.ip_data['lower_bound_ip_address'].apply(self.ip_to_integer)
        self.ip_data['upper_bound_ip_address'] = self.ip_data['upper_bound_ip_address'].apply(self.ip_to_integer)
        
        # Merging based on IP range
        merged_df = pd.merge_asof(
            self.fraud_data.sort_values('ip_address'),
            self.ip_data.sort_values('lower_bound_ip_address'),
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='forward'
        )
        
        self.fraud_data = merged_df

    
    @staticmethod
    def ip_to_integer(ip):
        # Convert IP to integer for merging
        parts = ip.split('.')
        return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
    
    def feature_engineering(self):
        # Transaction frequency and velocity
        self.fraud_data['signup_purchase_diff'] = (self.fraud_data['purchase_time'] - self.fraud_data['signup_time']).dt.total_seconds()
        
        # Hour and day of the week features
        self.fraud_data['hour_of_day'] = self.fraud_data['purchase_time'].dt.hour
        self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek
    
    def normalize_and_scale(self):
        # Scaling numeric columns
        scaler = StandardScaler()
        self.credit_data['Amount'] = scaler.fit_transform(self.credit_data[['Amount']])
        
        minmax_scaler = MinMaxScaler()
        self.fraud_data['purchase_value'] = minmax_scaler.fit_transform(self.fraud_data[['purchase_value']])
    
    def encode_categorical_features(self):
        # Encoding categorical features
        le = LabelEncoder()
        self.fraud_data['source'] = le.fit_transform(self.fraud_data['source'])
        self.fraud_data['browser'] = le.fit_transform(self.fraud_data['browser'])
        self.fraud_data['sex'] = le.fit_transform(self.fraud_data['sex'])
    
    def preprocess_all(self):
        # Run all preprocessing steps
        self.handle_missing_values()
        self.data_cleaning()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.merge_datasets()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        return self.fraud_data, self.credit_data
