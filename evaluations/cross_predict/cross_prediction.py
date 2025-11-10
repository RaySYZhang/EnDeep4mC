import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================== Configs ==================
# Species classification
SPECIES_CATEGORIES = {
    'Plant': {
        'color': 'green',
        'datasets': ['4mC_A.thaliana2', '4mC_C.equisetifolia', '4mC_F.vesca']
    },
    'Animal': {
        'color': 'blue',
        'datasets': ['4mC_C.elegans2', '4mC_D.melanogaster2']
    },
    'Microbe': {
        'color': 'red',
        'datasets': ['4mC_E.coli2', '4mC_G.subterraneus2', 
                    '4mC_G.pickeringii2', '4mC_S.cerevisiae', '4mC_Tolypocladium']
    }
}

CATEGORY_ORDER = ['Plant', 'Animal', 'Microbe']
ORDERED_DATASETS = []
for category in CATEGORY_ORDER:
    ORDERED_DATASETS.extend(SPECIES_CATEGORIES[category]['datasets'])

# ================== Original configs ==================
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/4mC')
MODEL_DIR = os.path.join(PROJECT_DIR, 'train_data_auto_indiv_extra')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'cross_predict')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(PROJECT_DIR)
from prepare.prepare_ml import ml_code, read_fasta_data
from lazy_feature_selection_extra import load_top_features, get_feature_methods

# ================== Universal function ==================
def load_best_n(model_name, dataset):
    """Load the optimal number of features"""
    acc_table_path = os.path.join(PROJECT_DIR, "extra_feature_ranking", 
                                 f"{model_name}_Feature_Acc_Table.csv")
    df = pd.read_csv(acc_table_path)
    best_n = df[df['N'] == 'best_n'][dataset].values[0]
    return int(float(best_n))

def prepare_source_data(source_dataset, model_type):
    """Prepare training data for the source species"""
    # Load training data
    train_pos = os.path.join(DATA_DIR, source_dataset, "train_pos.txt")
    train_neg = os.path.join(DATA_DIR, source_dataset, "train_neg.txt")
    
    # create dataFrame
    df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(train_pos)) + [0]*len(read_fasta_data(train_neg)),
        "seq": read_fasta_data(train_pos) + read_fasta_data(train_neg)
    })
    
    # Obtain feature configuration
    best_n = load_best_n(model_type, source_dataset)
    features = load_top_features(model_type, source_dataset, best_n)
    feature_methods = get_feature_methods(features)
    
    # generate features
    X_train, y_train, _ = ml_code(df, "training", feature_methods)
    return X_train, y_train, feature_methods

def prepare_target_data(target_dataset, feature_methods):
    """Prepare target test data"""

    test_pos = os.path.join(DATA_DIR, target_dataset, "test_pos.txt")
    test_neg = os.path.join(DATA_DIR, target_dataset, "test_neg.txt")
    
    df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(test_pos)) + [0]*len(read_fasta_data(test_neg)),
        "seq": read_fasta_data(test_pos) + read_fasta_data(test_neg)
    })
    
    X_test, y_test, _ = ml_code(df, "testing", feature_methods)
    return X_test, y_test

def predict_single_case(source, target):
    """Perform single cross prediction"""
    try:
        # Load ensemble model
        ensemble_path = os.path.join(MODEL_DIR, f'ensemble_indiv_{source}.pkl')
        ensemble_model = joblib.load(ensemble_path)
        
        meta_features = []
        y_target = None
        
        for model_type in BASE_MODELS:

            X_source_train, _, feature_methods = prepare_source_data(source, model_type)
            
            X_target, y_target = prepare_target_data(target, feature_methods)
            
            scaler = StandardScaler().fit(X_source_train)
            X_target_scaled = scaler.transform(X_target)
            
            model_path = os.path.join(MODEL_DIR, f'{model_type.lower()}_{source}.h5')
            base_model = tf.keras.models.load_model(model_path)

            input_data = X_target_scaled.reshape(-1, 1, X_target_scaled.shape[1])
            preds = base_model.predict(input_data, verbose=0).flatten()
            meta_features.append(preds)
        
        # Composite elemental features
        meta_X = np.column_stack(meta_features)
        
        # Integrated prediction
        y_pred = ensemble_model.predict(meta_X)
        if hasattr(ensemble_model, 'predict_proba'):
            y_proba = ensemble_model.predict_proba(meta_X)[:, 1]
        else:
            y_proba = y_pred
        
        return accuracy_score(y_target, y_pred), roc_auc_score(y_target, y_proba)
    
    except Exception as e:
        print(f"Error in {source}->{target}: {str(e)}")
        return np.nan, np.nan

# ================== Visualization ==================
def generate_heatmap(matrix, metric_name):

    plt.figure(figsize=(18, 15))
    ax = sns.heatmap(
        matrix.astype(float), 
        annot=True, 
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={'label': metric_name},
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    
    plt.title(f'Cross-Dataset {metric_name}', fontsize=16, pad=25)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.xlabel('Test Dataset', fontsize=14, labelpad=15)
    plt.ylabel('Train Dataset', fontsize=14, labelpad=15)

    ax.set_xticks(np.arange(len(ORDERED_DATASETS)) + 0.5)
    ax.set_xticklabels(ORDERED_DATASETS, rotation=45, ha='left', fontsize=10)
    ax.set_yticks(np.arange(len(ORDERED_DATASETS)) + 0.5)
    ax.set_yticklabels(ORDERED_DATASETS, rotation=0, fontsize=10)
    
    # Set classification label color
    def set_label_colors(labels, axis='x'):
        for label in labels:
            text = label.get_text()
            for category, info in SPECIES_CATEGORIES.items():
                if text in info['datasets']:
                    label.set_color(info['color'])
                    label.set_fontweight('bold')
                    if axis == 'x':
                        label.set_rotation(30)
                    break
    
    set_label_colors(ax.get_xticklabels(), 'x')
    set_label_colors(ax.get_yticklabels(), 'y')
    
    def draw_category_lines():
        accum_idx = 0
        for category in CATEGORY_ORDER:
            n = len(SPECIES_CATEGORIES[category]['datasets'])
            accum_idx += n
            ax.axhline(y=accum_idx, color='black', linewidth=2)
            ax.axvline(x=accum_idx, color='black', linewidth=2)
    
    draw_category_lines()
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, f'{metric_name.lower()}_heatmap.png')
    csv_path = os.path.join(OUTPUT_DIR, f'{metric_name.lower()}_matrix.csv')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save sorted CSV
    ordered_matrix = matrix.reindex(index=ORDERED_DATASETS, columns=ORDERED_DATASETS)
    ordered_matrix.to_csv(csv_path, float_format='%.4f')

# ================== Main ==================
def main():
    # Initialize result matrix (using ordered species list)
    acc_matrix = pd.DataFrame(index=ORDERED_DATASETS, columns=ORDERED_DATASETS, dtype=float)
    auc_matrix = pd.DataFrame(index=ORDERED_DATASETS, columns=ORDERED_DATASETS, dtype=float)

    # parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for source in ORDERED_DATASETS:
            for target in ORDERED_DATASETS:
                futures.append(executor.submit(predict_single_case, source, target))

        # Fill in results
        progress = tqdm(total=len(futures), desc="Processing cross-predictions")
        for i, future in enumerate(futures):
            source_idx = i // len(ORDERED_DATASETS)
            target_idx = i % len(ORDERED_DATASETS)
            source = ORDERED_DATASETS[source_idx]
            target = ORDERED_DATASETS[target_idx]
            
            acc, auc = future.result()
            acc_matrix.loc[source, target] = acc
            auc_matrix.loc[source, target] = auc
            progress.update()
        progress.close()

    generate_heatmap(acc_matrix, 'Accuracy')
    generate_heatmap(auc_matrix, 'AUC')
    print("Results have been saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()