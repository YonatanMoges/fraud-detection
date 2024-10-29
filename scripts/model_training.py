# scripts/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import mlflow
import mlflow.sklearn
import mlflow.keras
import joblib  # for local model saving
import os  # For directory management

class FraudDetectionModels:
    def __init__(self, fraud_data, credit_data):
        self.fraud_data = fraud_data
        self.credit_data = credit_data
        
    def prepare_data(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    
    def log_metrics(self, metrics):
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
    
    def save_model_locally(self, model, model_name, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(model, f"{directory}/{model_name}.joblib")
    
    def logistic_regression(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        if save_local:
            self.save_model_locally(model, "logistic_regression_model", directory)
        return metrics
    
    def decision_tree(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.sklearn.log_model(model, "decision_tree_model")
        if save_local:
            self.save_model_locally(model, "decision_tree_model", directory)
        return metrics
    
    def random_forest(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.sklearn.log_model(model, "random_forest_model")
        if save_local:
            self.save_model_locally(model, "random_forest_model", directory)
        return metrics
    
    def gradient_boosting(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        if save_local:
            self.save_model_locally(model, "gradient_boosting_model", directory)
        return metrics
    
    def mlp(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.sklearn.log_model(model, "mlp_model")
        if save_local:
            self.save_model_locally(model, "mlp_model", directory)
        return metrics
    
    def cnn(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = Sequential([
            Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.keras.log_model(model, "cnn_model")
        if save_local:
            os.makedirs(directory, exist_ok=True)
            model.save(f"{directory}/cnn_model.h5")  # saving in h5 format
        return metrics
    
    def rnn(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.keras.log_model(model, "rnn_model")
        if save_local:
            os.makedirs(directory, exist_ok=True)
            model.save(f"{directory}/rnn_model.h5")
        return metrics
    
    def lstm(self, X_train, X_test, y_train, y_test, save_local=False, directory="models"):
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        metrics = self.evaluate_model(model, X_test, y_test)
        mlflow.keras.log_model(model, "lstm_model")
        if save_local:
            os.makedirs(directory, exist_ok=True)
            model.save(f"{directory}/lstm_model.h5")
        return metrics
    
    def run_all_models(self, dataset, target_col, save_local=False, dataset_type="fraud"):
        X_train, X_test, y_train, y_test = self.prepare_data(dataset, target_col)
        
        # Set the directory based on dataset type
        model_directory = f"../models/{dataset_type}"

        results = {
            "Logistic Regression": self.logistic_regression(X_train, X_test, y_train, y_test, save_local, model_directory),
            "Decision Tree": self.decision_tree(X_train, X_test, y_train, y_test, save_local, model_directory),
            "Random Forest": self.random_forest(X_train, X_test, y_train, y_test, save_local, model_directory),
            "Gradient Boosting": self.gradient_boosting(X_train, X_test, y_train, y_test, save_local, model_directory),
            "MLP": self.mlp(X_train, X_test, y_train, y_test, save_local, model_directory),
            "CNN": self.cnn(X_train, X_test, y_train, y_test, save_local, model_directory),
            "RNN": self.rnn(X_train, X_test, y_train, y_test, save_local, model_directory),
            "LSTM": self.lstm(X_train, X_test, y_train, y_test, save_local, model_directory),
        }

        return results
