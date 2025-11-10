import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n
from models.deep_models import CNNModel, BLSTMModel, TransformerModel

# 配置参数
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
ALL_FEATURES = ['ENAC', 'binary', 'NCP', 'EIIP', 'Kmer4', 'CKSNAP', 'PseEIIP', 'TNC', 'RCKmer5', 'SCPseTNC', 'PCPseTNC', 'ANF', 'NAC', 'TAC']
DATASETS = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']

#BASE_MODELS = ['CNN']
#ALL_FEATURES = ['ENAC', 'binary', 'NCP', 'EIIP', 'Kmer4', 'CKSNAP', 'PseEIIP', 'TNC', 'RCKmer5', 'SCPseTNC', 'PCPseTNC', 'ANF', 'NAC', 'TAC']
#DATASETS = ['4mC_G.pickeringii']

SAVE_DIR = os.path.join(project_dir, 'evaluations', 'ablation_feature')
PRETRAIN_DIR = os.path.join(project_dir, 'pretrained_models/indiv')
os.makedirs(SAVE_DIR, exist_ok=True)

# result file mapping
PRETRAIN_FILES = {
    'CNN': 'cnn_auto_summary.csv',
    'BLSTM': 'blstm_auto_summary.csv',
    'Transformer': 'transformer_auto_summary.csv'
}

def load_pretrained_acc(model_type):
    """load accuracy for pre-trained models"""
    file_path = os.path.join(PRETRAIN_DIR, PRETRAIN_FILES[model_type])
    df = pd.read_csv(file_path)
    return df.set_index('Dataset')[f'{model_type}_Test_ACC'].to_dict()

# base model training function
def train_model_with_features(model_type, dataset, feature_methods):
    # data preparation
    base_dir = '/your_path/EnDeep4mC/data/4mC'
    train_pos = os.path.join(base_dir, dataset, "train_pos.txt")
    train_neg = os.path.join(base_dir, dataset, "train_neg.txt")
    test_pos = os.path.join(base_dir, dataset, "test_pos.txt")
    test_neg = os.path.join(base_dir, dataset, "test_neg.txt")

    train_df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(train_pos)) + [0]*len(read_fasta_data(train_neg)),
        "seq": read_fasta_data(train_pos) + read_fasta_data(train_neg)
    })
    test_df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(test_pos)) + [0]*len(read_fasta_data(test_neg)),
        "seq": read_fasta_data(test_pos) + read_fasta_data(test_neg)
    })

    # feature engineering
    X_train, y_train, _ = ml_code(train_df, "training", feature_methods)
    X_test, y_test, _ = ml_code(test_df, "testing", feature_methods)
    
    # data preprocessing
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # adjust input shape
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_val = X_val.reshape(-1, 1, X_val.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])

    # initialization
    if model_type == 'CNN':
        model = CNNModel(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            epochs=100,
            batch_size=512,
            dropout_rate=0.3,
            verbose=0
        )
    elif model_type == 'BLSTM':
        model = BLSTMModel(
            input_shape=(1, X_train.shape[2]),
            epochs=100,
            batch_size=512,
            dropout_rate=0.2,
            verbose=0
        )
    elif model_type == 'Transformer':
        model = TransformerModel(
            input_dim=X_train.shape[2],
            epochs=100,
            batch_size=512,
            dropout_rate=0.1,
            verbose=0
        )
    
    # training call
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# ablation experiment
def run_ablation_experiment(model_type):
    """ablation experiment(using pre-trained benchmarks)"""
    results = {dataset: {} for dataset in DATASETS}
    pretrain_acc = load_pretrained_acc(model_type)
    
    for dataset in tqdm(DATASETS, desc=f"Processing {model_type}"):
        try:
            base_acc = pretrain_acc[dataset]
            best_n = load_best_n(model_type, dataset)
            orig_features = load_top_features(model_type, dataset, best_n)
            
            for feature in tqdm(orig_features, desc=f"Ablating {dataset}", leave=False):
                ablated_features = [f for f in orig_features if f != feature]
                ablated_acc = train_model_with_features(model_type, dataset, get_feature_methods(ablated_features))

                delta = ablated_acc - base_acc
                results[dataset][feature] = delta
                
                print(f"\n特征消融: {feature}")
                print(f"基准准确率: {base_acc:.4f} | 消融后准确率: {ablated_acc:.4f}")
                print(f"差异(delta): {delta:+.4f} ({'提升' if delta > 0 else '下降'})")  # 添加符号显示
                print("-"*50)
                
        except Exception as e:
            print(f"Error in {model_type}-{dataset}: {str(e)}")
            continue
    
    result_df = pd.DataFrame(results).T
    result_df.to_csv(os.path.join(SAVE_DIR, f'ablation_{model_type}.csv'))
    return result_df

# visualization
def plot_ablation_results(model_type):
    result_path = os.path.join(SAVE_DIR, f'ablation_{model_type}.csv')
    df = pd.read_csv(result_path, index_col=0)
    
    plt.figure(figsize=(16, 10))
    for i, dataset in enumerate(DATASETS):
        plt.subplot(2, 3, i+1)
        
        try:
            data = df.loc[dataset].dropna().sort_values(ascending=False)
            filtered = data[np.abs(data) > 0.001]
            data = filtered if not filtered.empty else data.head(1)
        except KeyError:
            continue
            
        if data.empty:
            plt.text(0.5, 0.5, 'No Significant Features', 
                    ha='center', va='center')
            plt.axis('off')
            continue
            
        # Negative delta red (degradation), positive delta blue (improvement)
        colors = ['#2c7bb6' if x > 0 else '#d7191c' for x in data]
        
        # draw the horizontal bar chart
        plt.barh(data.index, data.values, color=colors)
        plt.title(dataset, fontsize=10)
        plt.xlabel('Feature Ablation Impact vs Best Features', fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.suptitle(f'{model_type} Model: Feature Ablation Impact Analysis', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'ablation_{model_type}.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    for model in BASE_MODELS:
        print(f"\n=== Running ablation for {model} ===")
        run_ablation_experiment(model)
        plot_ablation_results(model)
    
    # ensemble analysis
    ensemble_results = {}
    for dataset in DATASETS:
        delta_dict = {}
        for feature in ALL_FEATURES:
            total_delta = 0
            count = 0
            for model in BASE_MODELS:
                model_df = pd.read_csv(os.path.join(SAVE_DIR, f'ablation_{model}.csv'), index_col=0)
                if feature in model_df.columns:
                    # maintain the original delta direction during accumulation
                    total_delta += model_df.loc[dataset, feature]  
                    count += 1
            delta_dict[feature] = total_delta / count if count > 0 else 0
        ensemble_results[dataset] = delta_dict
    
    ensemble_df = pd.DataFrame(ensemble_results).T
    ensemble_df.to_csv(os.path.join(SAVE_DIR, 'ablation_Ensemble.csv'))
    
    # plot
    plt.figure(figsize=(16, 10))
    for i, dataset in enumerate(DATASETS):
        plt.subplot(2, 3, i+1)
        try:
            data = ensemble_df.loc[dataset].dropna().sort_values(ascending=False)
            filtered = data[np.abs(data) > 0.001]
            data = filtered if not filtered.empty else data.head(1)
        except KeyError:
            continue
            
        if data.empty:
            plt.text(0.5, 0.5, 'No Significant Features', 
                    ha='center', va='center')
            plt.axis('off')
            continue
            
        colors = ['#2c7bb6' if x > 0 else '#d7191c' for x in data]
        plt.barh(data.index, data.values, color=colors)
        plt.title(dataset, fontsize=10)
        plt.xlabel('Average Ablation Impact', fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.suptitle('Ensemble Model: Feature Ablation Impact Analysis', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'ablation_Ensemble.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()