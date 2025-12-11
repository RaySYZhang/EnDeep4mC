"""
EnDeep4mC - A Ensemble Deep Model for DNA 4mC Site Prediction
This module provides a unified interface for both 5-fold CV and independent test set models.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import warnings

# Configure GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

warnings.filterwarnings("ignore")

# Set up project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n, get_feature_names


class EnDeep4mC:
    """Unified EnDeep4mC model for both 5-fold CV and independent test set variants"""
    
    def __init__(self, dataset_name, model_type='5cv'):
        """
        Initialize the unified model
        
        Args:
            dataset_name (str): Name of the dataset (e.g., '4mC_G.pickeringii')
            model_type (str): Type of model to use - '5cv' or 'indiv'
        """
        self.base_models = ['CNN', 'BLSTM', 'Transformer']
        self.dataset_name = dataset_name
        self.model_type = model_type  # '5cv' or 'indiv'
        self.base_model_instances = {}
        self.scalers = {}
        self.feature_configs = {}
        self.meta_model = None
        self._load_models()
    
    def _load_models(self):
        """Load all pretrained models and configurations"""
        # Define base paths based on model type
        if self.model_type == '5cv':
            model_dir = '5cv'
            base_model_prefix = 'best'
        else:  # 'indiv'
            model_dir = 'indiv'
            base_model_prefix = ''
        
        # Construct paths
        base_path = os.path.join(project_dir, 'pretrained_models', model_dir)
        
        # Load meta model (ensemble)
        if self.model_type == '5cv':
            meta_model_filename = f'ensemble_5cv_{self.dataset_name}.pkl'
        else:
            meta_model_filename = f'ensemble_indiv_{self.dataset_name}.pkl'
        
        meta_model_path = os.path.join(base_path, meta_model_filename)
        self.meta_model = joblib.load(meta_model_path)
        print(f"Loaded {self.model_type} ensemble model for {self.dataset_name}")
        
        # Load base models and configurations
        for model_name in self.base_models:
            # Load base deep learning model
            if self.model_type == '5cv':
                model_filename = f'{model_name.lower()}_{base_model_prefix}_{self.dataset_name}.h5'
            else:
                model_filename = f'{model_name.lower()}_{self.dataset_name}.h5'
            
            model_path = os.path.join(base_path, model_filename)
            self.base_model_instances[model_name] = tf.keras.models.load_model(model_path)
            
            # Load feature configuration
            config_path = os.path.join(
                base_path,
                'feature_configs',
                f'{model_name}_{self.dataset_name}_config.pkl'
            )
            self.feature_configs[model_name] = joblib.load(config_path)
            
            # Load scaler
            if self.model_type == '5cv':
                scaler_path = os.path.join(
                    base_path,
                    'scalers',
                    f'{model_name}_{self.dataset_name}_scaler.pkl'
                )
            else:
                scaler_path = os.path.join(
                    base_path,
                    f'{model_name}_{self.dataset_name}_scaler.pkl'
                )
            self.scalers[model_name] = joblib.load(scaler_path)
        
        print(f"All {self.model_type} models and configurations loaded successfully")
    
    def _prepare_single_data(self, model_name, sequences, labels=None):
        """
        Prepare data for a single base model
        
        Args:
            model_name (str): Name of the base model
            sequences (list): List of DNA sequences
            labels (list, optional): List of labels
            
        Returns:
            np.ndarray: Prepared and scaled features
        """
        # Create DataFrame
        if labels is None:
            labels = [0] * len(sequences)  # Default labels if not provided
        
        df = pd.DataFrame({"label": labels, "seq": sequences})
        
        # Extract features using saved configuration
        config = self.feature_configs[model_name]
        X, _, feature_names = ml_code(df, "training", config['feature_methods'])
        
        # Select features based on saved indices
        X_selected = X[:, config['feature_indices']]
        
        # Scale features
        X_scaled = self.scalers[model_name].transform(X_selected)
        
        return X_scaled
    
    def _generate_meta_features(self, model_name, X):
        """
        Generate meta-features using a pretrained base model
        
        Args:
            model_name (str): Name of the base model
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Meta-features (predictions)
        """
        X_reshaped = X.reshape(-1, 1, X.shape[1])
        return self.base_model_instances[model_name].predict(X_reshaped, verbose=0).flatten()
    
    def predict_proba(self, sequences):
        """
        Predict probabilities for input sequences
        
        Args:
            sequences (list): List of DNA sequences
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        meta_features = []
        
        for model_name in self.base_models:
            # Prepare data for each base model
            X = self._prepare_single_data(model_name, sequences)
            
            # Generate meta-features
            meta_feature = self._generate_meta_features(model_name, X)
            meta_features.append(meta_feature)
        
        # Combine meta-features
        meta_X = np.column_stack(meta_features)
        
        # Predict using ensemble model
        probabilities = self.meta_model.predict_proba(meta_X)[:, 1]
        
        return probabilities
    
    def predict(self, sequences, threshold=0.5):
        """
        Predict labels for input sequences
        
        Args:
            sequences (list): List of DNA sequences
            threshold (float): Classification threshold
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        probabilities = self.predict_proba(sequences)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions
    
    def evaluate(self, sequences, true_labels, threshold=0.5):
        """
        Evaluate model performance on given data
        
        Args:
            sequences (list): List of DNA sequences
            true_labels (list): True labels
            threshold (float): Classification threshold
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                                     f1_score, matthews_corrcoef, roc_curve)
        
        probabilities = self.predict_proba(sequences)
        predictions = (probabilities > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'sensitivity': recall_score(true_labels, predictions),
            'specificity': recall_score(true_labels, predictions, pos_label=0),
            'f1_score': f1_score(true_labels, predictions),
            'mcc': matthews_corrcoef(true_labels, predictions),
            'auc': roc_auc_score(true_labels, probabilities)
        }
        
        # Calculate ROC curve data
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance from the ensemble model
        
        Returns:
            dict: Dictionary containing feature importance information
        """
        # Get base model importances
        base_importances = {}
        for model_name in self.base_models:
            if hasattr(self.base_model_instances[model_name], 'feature_importances_'):
                base_importances[model_name] = self.base_model_instances[model_name].feature_importances_
        
        # Get meta model feature importance if available
        meta_importance = None
        if hasattr(self.meta_model, 'feature_importances_'):
            meta_importance = self.meta_model.feature_importances_
        elif hasattr(self.meta_model.final_estimator_, 'coef_'):
            meta_importance = np.abs(self.meta_model.final_estimator_.coef_[0])
        
        return {
            'base_model_importances': base_importances,
            'meta_model_importance': meta_importance
        }


def load_model(dataset_name, model_type='5cv'):
    """
    Convenience function to load a pretrained model
    
    Args:
        dataset_name (str): Name of the dataset
        model_type (str): Type of model to load - '5cv' or 'indiv'
        
    Returns:
        EnDeep4mC: Loaded model instance
    """
    return EnDeep4mC(dataset_name, model_type)


def predict_sequences(sequences, dataset_name, model_type='5cv', threshold=0.5):
    """
    Convenience function for quick predictions
    
    Args:
        sequences (list): List of DNA sequences
        dataset_name (str): Name of the dataset
        model_type (str): Type of model to use - '5cv' or 'indiv'
        threshold (float): Classification threshold
        
    Returns:
        tuple: (predictions, probabilities)
    """
    model = load_model(dataset_name, model_type)
    probabilities = model.predict_proba(sequences)
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities

# Test with a sample sequence
if __name__ == "__main__":
    # Example usage
    datasets = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', 
                '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
    
    test_sequence = ["ATCGATCGATCGATCGATCG"]  # Example DNA sequence
    
    # Test both model types
    for model_type in ['5cv', 'indiv']:
        print(f"\n=== Testing {model_type} models ===")
        for dataset in datasets:
            print(f"\nDataset: {dataset}")
            try:
                model = load_model(dataset, model_type)
                
                # Predict probabilities
                probs = model.predict_proba(test_sequence)
                preds = model.predict(test_sequence)
                
                print(f"Predicted probability: {probs[0]:.4f}")
                print(f"Predicted label: {preds[0]}")
                
            except Exception as e:
                print(f"Error loading model for {dataset}: {str(e)}")
                continue