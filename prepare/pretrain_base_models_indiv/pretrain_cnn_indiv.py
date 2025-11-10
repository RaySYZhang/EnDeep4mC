import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# 配置GPU设置 - 简化配置避免冲突
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 只设置显存按需增长，不指定具体设备
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"检测到 {len(gpus)} 块GPU，已启用显存按需增长")
    except RuntimeError as e:
        print(e)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from models.deep_models import CNNModel
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n

def load_and_encode_fasta_data(dataset, feature_methods=None):
    base_dir = '/your_path/EnDeep4mC/data/4mC'
    
    train_pos_path = os.path.join(base_dir, dataset, "train_pos.txt")
    train_neg_path = os.path.join(base_dir, dataset, "train_neg.txt")
    test_pos_path = os.path.join(base_dir, dataset, "test_pos.txt")
    test_neg_path = os.path.join(base_dir, dataset, "test_neg.txt")

    train_df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(train_pos_path)) + [0]*len(read_fasta_data(train_neg_path)),
        "seq": read_fasta_data(train_pos_path) + read_fasta_data(train_neg_path)
    })
    test_df = pd.DataFrame({
        "label": [1]*len(read_fasta_data(test_pos_path)) + [0]*len(read_fasta_data(test_neg_path)),
        "seq": read_fasta_data(test_pos_path) + read_fasta_data(test_neg_path)
    })

    X_train, y_train, _ = ml_code(train_df, "training", feature_methods)
    X_test, y_test, _ = ml_code(test_df, "testing", feature_methods)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.15, 
        stratify=y_train, 
        random_state=42
    )

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    
    np.random.seed(42)
    noise = 0.05 * np.random.normal(0, 1, X_res.shape)
    X_train = np.vstack([X_res, X_res + noise])
    y_train = np.concatenate([y_res, y_res])

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_val = X_val.reshape(-1, 1, X_val.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_cnn_model(cnn_model, X_train, X_val, X_test, y_train, y_val, y_test, num_epochs, dataset_name):
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {i: w for i, w in enumerate(class_weights)}

    model_path = f'/your_path/EnDeep4mC/pretrained_models/indiv/cnn_{dataset_name}.h5'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = cnn_model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    cnn_model.model.load_weights(model_path)
    y_pred = (cnn_model.model.predict(X_test) > 0.5).astype(int).flatten()
    y_proba = cnn_model.model.predict(X_test).flatten()

    return y_pred, y_proba, history

def specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp + 1e-7)

def save_roc_data(dataset, model_name, fpr, tpr, auc):
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'AUC': [auc] * len(fpr)
    })
    roc_file = os.path.join(project_dir, 'pretrained_models/indiv_roc', f'{model_name}_roc_{dataset}.csv')
    roc_df.to_csv(roc_file, index=False)

def plot_roc_curve(dataset, model_name, fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({dataset})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(project_dir, 'pretrained_models/indiv_roc', f'{model_name}_roc_{dataset}.png'))
    plt.close()

if __name__ == "__main__":
    
    fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']

    summary_data = []
    model_name = 'CNN'
    
    for dataset in fasta_dataset_names:
        print(f"\n=== Processing Dataset: {dataset} ===")
        
        try:
            best_n = load_best_n(model_name, dataset)
            top_features = load_top_features(model_name, dataset, top_n=best_n)
            feature_methods = get_feature_methods(top_features)
        except Exception as e:
            print(f"Feature selection failed: {str(e)}, using default features")
            feature_methods = None
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_encode_fasta_data(dataset, feature_methods)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        cnn = CNNModel(
            input_shape=input_shape,
            epochs=100,
            batch_size=512,
            dropout_rate=0.3,
            l2_reg=0.001
        )
        
        y_pred, y_proba, history = train_cnn_model(
            cnn, X_train, X_val, X_test, 
            y_train, y_val, y_test,
            num_epochs=100,
            dataset_name=dataset
        )
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        save_roc_data(dataset, model_name, fpr, tpr, auc)
        plot_roc_curve(dataset, model_name, fpr, tpr, auc)

        acc = accuracy_score(y_test, y_pred)
        sn = recall_score(y_test, y_pred)
        sp = specificity(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        dataset_info = {
            'Dataset': dataset,
            'CNN_Test_ACC': f'{acc:.6f}',
            'CNN_SN': f'{sn:.6f}',
            'CNN_SP': f'{sp:.6f}',
            'CNN_F1': f'{f1:.6f}',
            'CNN_MCC': f'{mcc:.6f}',
            'CNN_AUC': f'{auc:.6f}'
        }
        summary_data.append(dataset_info)

    summary_df = pd.DataFrame(summary_data)
    print("\nFinal Test Result:")
    print(summary_df)
    summary_df.to_csv('/your_path/EnDeep4mC/pretrained_models/indiv/cnn_auto_summary.csv', index=False)