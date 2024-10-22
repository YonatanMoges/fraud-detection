import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import mlflow
import mlflow.sklearn
import mlflow.keras

class FraudDetectionModels:
    def __init__(self, fraud_data, credit_data):
        self.fraud_data = fraud_data
        self.credit_data = credit_data
        
    def prepare_data(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return self.evaluate_model(model, X_test, y_test)
    
    def decision_tree(self, X_train, X_test, y_train, y_test):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return self.evaluate_model(model, X_test, y_test)
    
    def random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return self.evaluate_model(model, X_test, y_test)
    
    def gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        return self.evaluate_model(model, X_test, y_test)
    
    def mlp(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
        model.fit(X_train, y_train)
        return self.evaluate_model(model, X_test, y_test)
    
    def cnn(self, X_train, X_test, y_train, y_test):
        model = Sequential([
            Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        return self.evaluate_model(model, X_test, y_test)
    
    def rnn(self, X_train, X_test, y_train, y_test):
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        return self.evaluate_model(model, X_test, y_test)
    
    def lstm(self, X_train, X_test, y_train, y_test):
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        return self.evaluate_model(model, X_test, y_test)

    def run_all_models(self, dataset, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(dataset, target_col)

        results = {
            "Logistic Regression": self.logistic_regression(X_train, X_test, y_train, y_test),
            "Decision Tree": self.decision_tree(X_train, X_test, y_train, y_test),
            "Random Forest": self.random_forest(X_train, X_test, y_train, y_test),
            "Gradient Boosting": self.gradient_boosting(X_train, X_test, y_train, y_test),
            "MLP": self.mlp(X_train, X_test, y_train, y_test),
            "CNN": self.cnn(X_train, X_test, y_train, y_test),
            "RNN": self.rnn(X_train, X_test, y_train, y_test),
            "LSTM": self.lstm(X_train, X_test, y_train, y_test),
        }

        return results
