import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

# Import all feature encoding schemes
from prepare.prepare_ml import ml_code, read_fasta_data
from fs.encode1 import ENAC2, binary, NCP, EIIP, NAC, PseEIIP, ANF, CKSNAP8, Kmer4, TNC, RCKmer5, DNC
from fs.load_acc import ACC_encoding
from fs.load_pse import pse

# Define extended feature encoding methods
def TAC(fastas): return ACC_encoding(fastas, method='TAC', type1='DNA', lag=2)
def TCC(fastas): return ACC_encoding(fastas, method='TCC', type1='DNA', lag=2)
def TACC(fastas): return ACC_encoding(fastas, method='TACC', type1='DNA', lag=2)
def PseDNC(fastas): return pse(fastas, method='PseDNC', kmer=3)
def PseKNC(fastas): return pse(fastas, method='PseKNC', kmer=3)
def PCPseDNC(fastas): return pse(fastas, method='PCPseDNC', kmer=3)
def SCPseDNC(fastas): return pse(fastas, method='SCPseDNC', kmer=3)
def SCPseTNC(fastas): return pse(fastas, method='SCPseTNC', kmer=3)
def PCPseTNC(fastas): return pse(fastas, method='PCPseTNC', kmer=3)

def load_top_features(model_name, dataset_name, top_n=10):
    """Load feature sorting based on model name and dataset name"""
    rank_file = os.path.join(project_dir, "feature_engineering/fea_index", f"{model_name}_Ranking.csv")
    if not os.path.exists(rank_file):
        raise FileNotFoundError(f"Ranking file {rank_file} not found")
    
    df = pd.read_csv(rank_file, index_col=0)
    if dataset_name not in df.index:
        raise ValueError(f"Dataset {dataset_name} not found in {rank_file}")
    
    features = []
    for i in range(top_n):
        col_name = f"Rank_{i+1}"
        if col_name not in df.columns:
            break
        features.append(df.loc[dataset_name, col_name])
    
    return features[:top_n]

def load_best_n(model_name, dataset_name):
    """Load the optimal number of features"""
    acc_table_path = os.path.join(project_dir, "feature_engineering/ifs_result", f"{model_name}_Feature_Acc_Table.csv")
    if not os.path.exists(acc_table_path):
        raise FileNotFoundError(f"Accuracy table {acc_table_path} not found")
    
    acc_table = pd.read_csv(acc_table_path)
    best_n_row = acc_table[acc_table['N'] == 'best_n']
    
    if dataset_name not in best_n_row.columns:
        raise ValueError(f"Dataset {dataset_name} not found in accuracy table")
    
    return int(best_n_row[dataset_name].values[0])

def get_feature_methods(selected_features):
    """Map feature names to corresponding method functions"""
    # Directly use defined named functions
    all_methods = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP,
        "Kmer4": Kmer4,
        "CKSNAP": CKSNAP8,
        "PseEIIP": PseEIIP,
        "TNC": TNC,
        "RCKmer5": RCKmer5,
        "SCPseTNC": SCPseTNC,
        "PCPseTNC": PCPseTNC,
        "ANF": ANF,
        "NAC": NAC,
        "TAC": TAC,
    }
    
    return {name: all_methods[name] for name in selected_features if name in all_methods}

def get_feature_names(selected_features):
    """Return a list of feature names instead of a function object dictionary"""
    valid_features = [
        "ENAC", "binary", "NCP", "EIIP", "Kmer4", 
        "CKSNAP", "PseEIIP", "TNC", "RCKmer5",
        "SCPseTNC", "PCPseTNC", "ANF", "NAC", "TAC"
    ]
    return [name for name in selected_features if name in valid_features]