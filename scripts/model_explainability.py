import shap
import lime
import lime.lime_tabular
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans

class ModelExplainability:
    def __init__(self, model_paths):
        # Dictionary containing model names and paths
        self.model_paths = model_paths
        self.models = self.load_models()
    
    def load_models(self):
        models = {}
        for model_name, path in self.model_paths.items():
            if path.endswith(".joblib"):
                models[model_name] = joblib.load(path)
            elif path.endswith(".h5"):
                models[model_name] = tf.keras.models.load_model(path)
            else:
                raise ValueError(f"Unsupported model file format: {path}")
        return models

    

    def get_shap_explainer(self, model, X_train, num_clusters=50):
        # Use TreeExplainer for tree-based models and KernelExplainer otherwise
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            return shap.TreeExplainer(model)
        else:
            # Cluster the data using k-means to get representative points
            background = shap.kmeans(X_train, num_clusters)
            return shap.KernelExplainer(model.predict, background)

    def shap_summary_plot(self, model_name, X_train, num_clusters=50):
        model = self.models[model_name]
        # Generate explainer with k-means clustered background data
        explainer = self.get_shap_explainer(model, X_train, num_clusters=num_clusters)
        
        # Compute SHAP values with clustered background data
        shap_values = explainer.shap_values(X_train)
        
        # Check SHAP values shape
        print("SHAP values shape:", shap_values.shape if not isinstance(shap_values, list) else [sv.shape for sv in shap_values])
        print("X_train shape:", X_train.shape)
        
        plt.figure()
        shap.summary_plot(shap_values, X_train)
        plt.title(f'SHAP Summary Plot for {model_name} with Clustered Background')
        plt.show()

   

    def shap_force_plot(self, model_name, X_train, index):
        model = self.models[model_name]
        explainer = self.get_shap_explainer(model, X_train)
        shap_values = explainer.shap_values(X_train)

        # Handling multi-output models
        if isinstance(shap_values, list):  # Multi-output model
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            shap_value = shap_values[1][index] if len(shap_values) > 1 else shap_values[0][index]
        else:  # Single-output model
            base_value = explainer.expected_value
            shap_value = shap_values[index]

        # Updated force plot call with `matplotlib=True` for inline plots in some setups
        shap.plots.force(base_value, shap_value, X_train.iloc[index, :], matplotlib=True)


    
    def shap_dependence_plot(self, model_name, X_train, feature):
        model = self.models[model_name]
        explainer = self.get_shap_explainer(model, X_train)
        shap_values = explainer.shap_values(X_train)
        
        plt.figure()
        shap.dependence_plot(feature, shap_values[1] if isinstance(shap_values, list) else shap_values, X_train)
        plt.title(f'SHAP Dependence Plot for {model_name} - Feature: {feature}')
        plt.show()

    def lime_explanation(self, model_name, X_train, index):
        model = self.models[model_name]
        
        # Define the explainer mode based on model type
        mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            mode=mode,
            feature_names=X_train.columns.tolist(),
            class_names=['No Fraud', 'Fraud'] if mode == 'classification' else None
        )

        # Use predict_proba for classifiers, else predict
        prediction_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        exp = explainer.explain_instance(X_train.iloc[index, :], prediction_fn)
        exp.show_in_notebook(show_all=False)
