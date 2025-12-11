"""
Training script for Independent Test Set Ensemble Model
This script trains an ensemble model using independent training and testing sets.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, matthews_corrcoef, roc_curve)
from lightgbm import LGBMClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import warnings

# Configure GPU settings - Simplified to avoid conflicts
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth only, do not specify specific device
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

warnings.filterwarnings("ignore")

# Set up project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n, get_feature_names


class PretrainEnsembleIndiv:
    """Training class for Independent Test Set Ensemble Model"""
    
    def __init__(self):
        """Initialize the training model"""
        self.base_models = ['CNN', 'BLSTM', 'Transformer']
        self.scalers = {}
        self.meta_model = None
        self.dataset_name = None
        self.feature_configs = {}
        self.best_meta_model = None
        self.full_data_cache = {}

    def _get_best_n_features(self, model_name):
        """
        Get the optimal number of features for a specific model
        
        Args:
            model_name (str): Name of the base model
            
        Returns:
            int: Number of best features to use
        """
        try:
            return load_best_n(model_name, self.dataset_name)
        except Exception as e:
            print(f"Loading optimal features failed: {str(e)}, using default value 10")
            return 10

    def _load_full_dataset(self, model_name):
        """
        Load and merge all data for unified feature engineering
        
        Args:
            model_name (str): Name of the base model
            
        Returns:
            tuple: (X_full, y_full) - Complete feature matrix and labels
        """
        if (model_name, self.dataset_name) in self.full_data_cache:
            return self.full_data_cache[(model_name, self.dataset_name)]

        base_dir = os.path.join(project_dir, 'data', '4mC')
        paths = {
            'train_pos': os.path.join(base_dir, self.dataset_name, "train_pos.txt"),
            'train_neg': os.path.join(base_dir, self.dataset_name, "train_neg.txt"),
            'test_pos': os.path.join(base_dir, self.dataset_name, "test_pos.txt"),
            'test_neg': os.path.join(base_dir, self.dataset_name, "test_neg.txt")
        }

        # Merge all original data
        full_df = pd.DataFrame({
            "label": ([1]*len(read_fasta_data(paths['train_pos'])) + 
                     [0]*len(read_fasta_data(paths['train_neg'])) +
                     [1]*len(read_fasta_data(paths['test_pos'])) + 
                     [0]*len(read_fasta_data(paths['test_neg']))),
            "seq": (read_fasta_data(paths['train_pos']) + 
                   read_fasta_data(paths['train_neg']) +
                   read_fasta_data(paths['test_pos']) + 
                   read_fasta_data(paths['test_neg']))
        })

        # Unified Feature Engineering
        best_n = self._get_best_n_features(model_name)
        self.feature_configs[model_name] = get_feature_methods(
            load_top_features(model_name, self.dataset_name, best_n)
        )

        # Obtain feature names and indexes
        X_full, y_full, feature_names = ml_code(full_df, "training", self.feature_configs[model_name])
        selected_features = load_top_features(model_name, self.dataset_name, best_n)
        feature_list = get_feature_names(selected_features)  # Get a list of names

        # Generate feature index
        feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]

        # Add feature configuration and save
        config_dir = os.path.join(project_dir, 'pretrained_models', 'indiv', 'feature_configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f'{model_name}_{self.dataset_name}_config.pkl')
        joblib.dump({
            'selected_features': feature_list,
            'feature_methods': self.feature_configs[model_name],
            'feature_indices': feature_indices
        }, config_path)

        # Cache processing
        self.full_data_cache[(model_name, self.dataset_name)] = (X_full, y_full)
        return X_full, y_full

    def _prepare_data(self, model_name):
        """
        Data preprocessing process
        
        Args:
            model_name (str): Name of the base model
            
        Returns:
            tuple: (X_train_aug, y_train_aug, X_test_scaled, y_test) - Processed datasets
        """
        # Load full data
        X_full, y_full = self._load_full_dataset(model_name)
        
        # Calculate the sample size of the original training set
        base_dir = os.path.join(project_dir, 'data', '4mC')
        train_size = (
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_pos.txt"))) +
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_neg.txt")))
        )
        
        # Safe data separation
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        X_test = X_full[train_size:]
        y_test = y_full[train_size:]

        # Data augmentation is only applied to the training set
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        
        # Standardization
        self.scalers[model_name] = StandardScaler()
        X_res_scaled = self.scalers[model_name].fit_transform(X_res)
        
        # Add noise enhancement
        np.random.seed(42)
        noise = 0.1 * np.random.normal(0, 1, X_res_scaled.shape)
        X_train_aug = np.vstack([X_res_scaled, X_res_scaled + noise])
        y_train_aug = np.concatenate([y_res, y_res])
        
        # Process test data
        X_test_scaled = self.scalers[model_name].transform(X_test)
        
        # Save standardization tool
        scaler_dir = os.path.join(project_dir, 'pretrained_models', 'indiv')
        scaler_path = os.path.join(scaler_dir, f'{model_name}_{self.dataset_name}_scaler.pkl')
        joblib.dump(self.scalers[model_name], scaler_path)
        
        return X_train_aug, y_train_aug, X_test_scaled, y_test

    def _load_pretrained_model(self, model_name):
        """
        Load a pretrained base model
        
        Args:
            model_name (str): Name of the base model
            
        Returns:
            tf.keras.Model: Loaded pretrained model
        """
        model_path = os.path.join(project_dir, 'pretrained_models', 'indiv', f'{model_name.lower()}_{self.dataset_name}.h5')
        return tf.keras.models.load_model(model_path)

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
        return self._load_pretrained_model(model_name).predict(X_reshaped, verbose=0).flatten()

    def _prepare_meta_dataset(self):
        """
        Prepare meta-dataset by combining predictions from all base models
        
        Returns:
            tuple: (meta_X_train, y_train_full, meta_X_test, y_test_full) - Meta-datasets
        """
        meta_train, meta_test = [], []
        y_train_full, y_test_full = None, None

        for model_name in tqdm(self.base_models, desc="Processing base models"):
            X_train, y_train, X_test, y_test = self._prepare_data(model_name)
            
            # Generate meta feature
            train_feat = self._generate_meta_features(model_name, X_train)
            test_feat = self._generate_meta_features(model_name, X_test)
            
            meta_train.append(train_feat)
            meta_test.append(test_feat)
            
            if y_train_full is None:
                y_train_full = y_train
                y_test_full = y_test
            else:
                assert np.array_equal(y_train_full, y_train), "Inconsistent training labels"
                assert np.array_equal(y_test_full, y_test), "Inconsistent testing labels"

        return np.column_stack(meta_train), y_train_full, np.column_stack(meta_test), y_test_full

    def train_and_evaluate(self):
        """
        Train and evaluate the ensemble model
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Prepare metadata set
        meta_X_train, y_train, meta_X_test, y_test = self._prepare_meta_dataset()

        # Define base models for stacking
        base_models = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=20
            ))
        ]

        # Define final meta-model
        final_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=0.6,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )

        # Train the meta learner
        self.meta_model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_model,
            stack_method='predict_proba',
            passthrough=True,
            n_jobs=-1
        )
        self.meta_model.fit(meta_X_train, y_train)

        # Save model
        model_path = os.path.join(project_dir, 'pretrained_models', 'indiv', f'ensemble_indiv_{self.dataset_name}.pkl')
        joblib.dump(self.meta_model, model_path)
        print(f"Model saved to: {model_path}")

        # Evaluate performance
        y_proba = self.meta_model.predict_proba(meta_X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'sn': recall_score(y_test, y_pred),
            'sp': recall_score(y_test, y_pred, pos_label=0),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        # Save ROC data
        roc_dir = os.path.join(project_dir, 'pretrained_models', 'indiv_roc')
        os.makedirs(roc_dir, exist_ok=True)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'AUC': [metrics['auc']]*len(fpr)})
        roc_df.to_csv(
            os.path.join(roc_dir, f'Ensemble_roc_{self.dataset_name}.csv'),
            index=False,
            float_format='%.6f'
        )
        
        # Plot and save ROC curve
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble (AUC = {metrics["auc"]:.6f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({self.dataset_name})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_dir, f'Ensemble_roc_{self.dataset_name}.png'))
        plt.close()

        return metrics


def main():
    """Main execution function for training"""
    datasets = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', 
                '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']

    results = []
    
    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        try:
            ensemble = PretrainEnsembleIndiv()
            ensemble.dataset_name = dataset
            
            metrics = ensemble.train_and_evaluate()
            
            formatted_metrics = {k: f"{v:.6f}" for k, v in metrics.items()}
            results.append({
                'Dataset': dataset,
                **formatted_metrics
            })
            
            print(f"\n{dataset} Testing Results:")
            for k, v in formatted_metrics.items():
                print(f"{k.upper()}: {v}")
                
        except Exception as e:
            print(f"Failed to process {dataset}: {str(e)}")
            continue
    
    # Save final results
    result_df = pd.DataFrame(results)
    save_path = os.path.join(project_dir, 'pretrained_models', 'indiv', 'ensemble_indiv_results.csv')
    result_df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"\nFinal results saved to: {save_path}")


if __name__ == "__main__":
    main()