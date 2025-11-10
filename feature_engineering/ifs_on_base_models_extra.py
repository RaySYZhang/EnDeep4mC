import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

current_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(current_dir, "ifs_result_extra_species")
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.pretrain_base_models_indiv.pretrain_cnn_indiv import load_and_encode_fasta_data as cnn_load, CNNModel, train_cnn_model
from prepare.pretrain_base_models_indiv.pretrain_blstm_indiv import load_and_encode_fasta_data as blstm_load, BLSTMModel, train_blstm_model
from prepare.pretrain_base_models_indiv.pretrain_transformer_indiv import load_and_encode_fasta_data as transformer_load, TransformerModel, train_transformer_model
from feature_selection_extra import load_top_features, get_feature_methods

DATASETS = [
    '4mC_A.thaliana2', '4mC_C.elegans2', '4mC_C.equisetifolia', '4mC_D.melanogaster2', '4mC_E.coli2', '4mC_F.vesca', '4mC_G.pickeringii2', '4mC_G.subterraneus2', '4mC_S.cerevisiae', '4mC_Tolypocladium'
    #'4mC_A.thaliana2'
]

MODEL_CONFIGS = [
    {
        'name': 'CNN',
        'loader': cnn_load,
        'model_class': CNNModel,
        'trainer': train_cnn_model,
        'params': {'epochs': 50, 'batch_size': 512, 'dropout_rate': 0.3}
    },
    {
        'name': 'BLSTM',
        'loader': blstm_load,
        'model_class': BLSTMModel,
        'trainer': train_blstm_model,
        'params': {'epochs': 50, 'batch_size': 512, 'dropout_rate': 0.2}
    },
    {
        'name': 'Transformer',
        'loader': transformer_load,
        'model_class': TransformerModel,
        'trainer': train_transformer_model,
        'params': {'epochs': 50, 'batch_size': 512, 'dropout_rate': 0.1}
    }
]

COLORS = plt.cm.tab10.colors

def evaluate_model(config):
    model_name = config['name']
    print(f"\n=== Processing {model_name} model ===")

    results = {dataset: {'N': [], 'Accuracy': []} for dataset in DATASETS}
    
    for dataset in tqdm(DATASETS, desc="Datasets Progress"):
        try:
            top_features = load_top_features(model_name, dataset, top_n=14)
        except Exception as e:
            print(f"Cannot load {dataset}: {str(e)}")
            continue

        for n in tqdm(range(1, 15), desc=f"feature nums progress", leave=False):
            selected_features = top_features[:n]
            feature_methods = get_feature_methods(selected_features)
            
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = config['loader'](
                    dataset, feature_methods
                )

                if model_name in ['CNN', 'BLSTM']:
                    model = config['model_class'](
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        **config['params']
                    )
                else:
                    model = config['model_class'](
                        input_dim=X_train.shape[2],
                        **config['params']
                    )

                result = config['trainer'](
                    model, X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    num_epochs=config['params']['epochs'],
                    dataset_name=dataset
                )
                y_pred = result[0]

                acc = accuracy_score(y_test, y_pred)

                results[dataset]['N'].append(n)
                results[dataset]['Accuracy'].append(acc)
                
            except Exception as e:
                print(f"Error when processing {dataset} N={n}: {str(e)}")
                continue

    save_results(model_name, results)

    plot_results(model_name, results)

def save_results(model_name, results):

    df = pd.DataFrame(columns=['N'] + DATASETS)
    
    for n in range(1, 15):
        row = {'N': n}
        for dataset in DATASETS:
            try:
                idx = results[dataset]['N'].index(n)
                row[dataset] = results[dataset]['Accuracy'][idx]
            except:
                row[dataset] = np.nan
        df = df.append(row, ignore_index=True)

    df['N'] = df['N'].astype(int)

    best_row = {'N': 'best_n'}
    for dataset in DATASETS:
        n_list = results[dataset].get('N', [])
        acc_list = results[dataset].get('Accuracy', [])
        
        if not n_list or not acc_list:
            best_row[dataset] = np.nan
            continue

        max_acc = max(acc_list)
        best_n_candidates = [n for n, acc in zip(n_list, acc_list) if acc == max_acc]

        best_row[dataset] = int(min(best_n_candidates)) if best_n_candidates else np.nan
    
    df = df.append(best_row, ignore_index=True)

    output_path = os.path.join(SAVE_DIR, f"{model_name}_Feature_Acc_Table.csv")
    df.to_csv(output_path, index=False)
    print(f"{model_name} results have been saved to {output_path}")

def plot_results(model_name, results):
    plt.figure(figsize=(12, 6))
    
    for i, dataset in enumerate(DATASETS):
        if len(results[dataset]['N']) > 0:
            plt.plot(
                results[dataset]['N'],
                results[dataset]['Accuracy'],
                marker='o',
                color=COLORS[i],
                label=dataset
            )
    
    plt.xlabel('Number of Features (N)')
    plt.ylabel('Accuracy')
    plt.title(f'Feature Selection Accuracy - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(range(1, 15))

    output_path = os.path.join(SAVE_DIR, f"{model_name}_Feature_Acc_Curve.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"{model_name} curve graph have been saved to {output_path}")

if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    for config in MODEL_CONFIGS:
        evaluate_model(config)
    
    print("\nProcessing successfully! Please check the SAVE_DIR.")