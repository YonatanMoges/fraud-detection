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
        
        # Convert datetime fields if they exist in fraud_data
        if 'signup_time' in self.fraud_data.columns:
            self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'], errors='coerce')
        if 'purchase_time' in self.fraud_data.columns:
            self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'], errors='coerce')

    
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
        # Ensure IP format is valid (i.e., has 4 parts); if not, return a default integer (0)
        parts = ip.split('.')
        if len(parts) != 4:
            return 0  # Use a default integer for invalid IPs
        
        try:
            # Convert each part to an integer and calculate the full IP integer value
            return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
        except ValueError:
            return 0  # Return 0 if any part of IP address is not an integer

    
    def feature_engineering(self):
        # Check for datetime columns and ensure they are converted
        if 'signup_time' in self.fraud_data.columns:
            self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'], errors='coerce')
        if 'purchase_time' in self.fraud_data.columns:
            self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'], errors='coerce')
        
        # Only proceed with feature engineering if both columns are present
        if 'signup_time' in self.fraud_data.columns and 'purchase_time' in self.fraud_data.columns:
            # Drop rows where datetime conversion failed
            self.fraud_data.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
            
            # Extract features
            self.fraud_data['signup_purchase_diff'] = (self.fraud_data['purchase_time'] - self.fraud_data['signup_time']).dt.total_seconds()
            self.fraud_data['hour_of_day'] = self.fraud_data['purchase_time'].dt.hour
            self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek
            
            # Remove the original datetime columns to prevent issues during model training
            self.fraud_data.drop(columns=['signup_time', 'purchase_time'], inplace=True)

    def normalize_and_scale(self):
        # Scaling numeric columns (apply only on numeric columns)
        scaler = StandardScaler()
        num_columns = self.credit_data.select_dtypes(include=[np.number]).columns
        self.credit_data[num_columns] = scaler.fit_transform(self.credit_data[num_columns])

        minmax_scaler = MinMaxScaler()
        fraud_numeric_columns = self.fraud_data.select_dtypes(include=[np.number]).columns
        self.fraud_data[fraud_numeric_columns] = minmax_scaler.fit_transform(self.fraud_data[fraud_numeric_columns])

    def encode_categorical_features(self):
        # Automatically detect categorical columns and encode them
        categorical_columns = self.fraud_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            self.fraud_data[col] = le.fit_transform(self.fraud_data[col].astype(str))

        # Repeat for credit data if needed
        credit_categorical_columns = self.credit_data.select_dtypes(include=['object']).columns
        for col in credit_categorical_columns:
            le = LabelEncoder()
            self.credit_data[col] = le.fit_transform(self.credit_data[col].astype(str))

    
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
    
    def save_processed_data(self, fraud_output_path="processed_fraud_data.csv", credit_output_path="processed_credit_data.csv"):
        """
        Saves the preprocessed fraud and credit datasets to CSV files.
        
        Parameters:
        - fraud_output_path (str): File path to save the processed fraud data.
        - credit_output_path (str): File path to save the processed credit card data.
        """
        # Save the processed datasets to CSV
        self.fraud_data.to_csv(fraud_output_path, index=False)
        self.credit_data.to_csv(credit_output_path, index=False)
