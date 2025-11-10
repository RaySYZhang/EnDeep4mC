import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score

BASE_DIR = "/your_path/EnDeep4mC/data/4mC"
current_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(current_dir, "fea_index_extra_species")
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from fs.encode1 import ENAC2, binary, NCP, EIIP, NAC, PseEIIP, ANF, CKSNAP8, Kmer4, TNC, RCKmer5
from fs.load_acc import ACC_encoding
from fs.load_pse import pse

from prepare.pretrain_base_models_indiv.pretrain_cnn_indiv import load_and_encode_fasta_data as cnn_load, CNNModel, train_cnn_model
from prepare.pretrain_base_models_indiv.pretrain_blstm_indiv import load_and_encode_fasta_data as blstm_load, BLSTMModel, train_blstm_model
from prepare.pretrain_base_models_indiv.pretrain_transformer_indiv import load_and_encode_fasta_data as transformer_load, TransformerModel, train_transformer_model

FEATURE_METHODS = {
    "ENAC": ENAC2, "binary": binary, "NCP": NCP, "EIIP": EIIP, 
    "Kmer4": Kmer4, "CKSNAP": CKSNAP8, "PseEIIP": PseEIIP, 
    "TNC": TNC, "RCKmer5": RCKmer5, 
    "SCPseTNC": lambda x: pse(x, method='SCPseTNC', kmer=3),
    "PCPseTNC": lambda x: pse(x, method='PCPseTNC', kmer=3),
    "ANF": ANF, "NAC": NAC, 
    "TAC": lambda x: ACC_encoding(x, method='TAC', type1='DNA', lag=2),
}

DATASETS = [
    '4mC_A.thaliana2', '4mC_C.elegans2', '4mC_C.equisetifolia', '4mC_D.melanogaster2', '4mC_E.coli2', '4mC_F.vesca', '4mC_G.pickeringii2', '4mC_G.subterraneus2', '4mC_S.cerevisiae', '4mC_Tolypocladium'
    #'4mC_E.coli'
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

def evaluate_features():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for config in MODEL_CONFIGS:
        model_name = config['name']
        print(f"\n=== Processing {model_name} model ===")

        full_ranking = {}
        acc_matrix = {}

        for dataset in tqdm(DATASETS, desc="Dataset Progress"):
            dataset_acc = {}
            
            for feat_name, feat_method in tqdm(FEATURE_METHODS.items(), desc="feature evaluation", leave=False):
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test = config['loader'](
                        dataset, {feat_name: feat_method}
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
                except Exception as e:
                    print(f"Error in {feat_name}@{dataset}: {str(e)}")
                    acc = np.nan
                
                dataset_acc[feat_name] = acc

            sorted_features = sorted(dataset_acc.items(), key=lambda x: x[1], reverse=True)
            ranking = {f"Rank_{i+1}": feat[0] for i, feat in enumerate(sorted_features)}

            full_ranking[dataset] = ranking
            acc_matrix[dataset] = dataset_acc

        ranking_df = pd.DataFrame.from_dict(full_ranking, orient='index')
        ranking_df.to_csv(os.path.join(SAVE_DIR, f"{model_name}_Ranking.csv"))
        
        acc_df = pd.DataFrame.from_dict(acc_matrix, orient='index')
        acc_df.to_csv(os.path.join(SAVE_DIR, f"{model_name}_ACC_Matrix.csv"))

        print(f"{model_name} model processing completed! The result has been saved.")

if __name__ == "__main__":
    evaluate_features()
    print("\nAll feature sorting has been completed! Please check feature_engineering/fea_index_extra_species 目录")