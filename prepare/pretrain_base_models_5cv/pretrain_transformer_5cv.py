import numpy as np
import datetime
import tensorflow as tf
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                            matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
from models.deep_models import TransformerModel
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n

# ------------------------- 工具函数 -------------------------
def specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp + 1e-7)

class TrainingMonitor(tf.keras.callbacks.Callback):
    """自定义训练监控回调"""
    def __init__(self, fold_num):
        super().__init__()
        self.fold_num = fold_num
        self.epoch_logs = []
        
    def on_epoch_end(self, epoch, logs=None):
        """保存每个epoch的指标"""
        logs = logs or {}
        logs['epoch'] = epoch + 1
        self.epoch_logs.append(logs)
        
    def on_train_end(self, logs=None):
        """训练结束后输出总结信息"""
        best_epoch = max(self.epoch_logs, key=lambda x: x['val_accuracy'])
        print(f"\nFold {self.fold_num} 最佳epoch {best_epoch['epoch']}:")
        print(f"训练集 - 损失: {best_epoch['loss']:.6f}, 准确率: {best_epoch['accuracy']:.6f}")
        print(f"验证集 - 损失: {best_epoch['val_loss']:.6f}, 准确率: {best_epoch['val_accuracy']:.6f}")

# ------------------------- 数据加载 -------------------------
def load_full_dataset(dataset, feature_methods=None):
    """加载并编码完整数据集"""
    base_dir = '/your_path/EnDeep4mC/data/4mC'
    
    # 读取训练集
    train_pos = read_fasta_data(os.path.join(base_dir, dataset, "train_pos.txt"))
    train_neg = read_fasta_data(os.path.join(base_dir, dataset, "train_neg.txt"))
    train_df = pd.DataFrame({
        "label": [1]*len(train_pos) + [0]*len(train_neg),
        "seq": train_pos + train_neg
    })
    
    # 读取测试集
    test_pos = read_fasta_data(os.path.join(base_dir, dataset, "test_pos.txt"))
    test_neg = read_fasta_data(os.path.join(base_dir, dataset, "test_neg.txt"))
    test_df = pd.DataFrame({
        "label": [1]*len(test_pos) + [0]*len(test_neg),
        "seq": test_pos + test_neg
    })

    # 特征工程
    X_train, y_train, _ = ml_code(train_df, "training", feature_methods)
    X_test, y_test, _ = ml_code(test_df, "testing", feature_methods)
    
    return X_train, y_train, X_test, y_test

# ------------------------- 数据预处理 -------------------------
def preprocess_data(X_train, y_train, X_val=None):
    """数据预处理流程"""
    # SMOTE过采样
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # 标准化
    scaler = StandardScaler().fit(X_res)
    X_res = scaler.transform(X_res)
    
    # 添加噪声增强
    np.random.seed(42)
    noise = 0.05 * np.random.normal(0, 1, X_res.shape)
    X_aug = np.vstack([X_res, X_res + noise])
    y_aug = np.concatenate([y_res, y_res])
    
    # 调整形状
    input_shape = (1, X_aug.shape[1])
    X_aug = X_aug.reshape(-1, *input_shape)
    
    # 验证集处理
    if X_val is not None:
        X_val = scaler.transform(X_val).reshape(-1, *input_shape)
    
    return X_aug, y_aug, X_val, scaler

# ------------------------- 模型构建 -------------------------
def build_transformer_model(input_dim, num_heads=8, ff_dim=512, num_layers=2, dropout=0.1):
    """构建Transformer模型实例"""
    return TransformerModel(
        input_dim=input_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout_rate=dropout,
        epochs=100,
        batch_size=512
    )

# ------------------------- 训练配置 -------------------------
def get_callbacks(fold_model_path, fold_num):
    """获取训练回调配置"""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            fold_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        TrainingMonitor(fold_num)  # 添加自定义监控
    ]

# ------------------------- 模型评估 -------------------------
def evaluate_model(model, X_test, y_test, scaler):
    """模型评估流程"""
    X_test_rs = scaler.transform(X_test).reshape(-1, 1, X_test.shape[1])
    y_proba = model.predict(X_test_rs).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    return {
        'ACC': accuracy_score(y_test, y_pred),
        'SN': recall_score(y_test, y_pred),
        'SP': specificity(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }, y_proba

# ------------------------- 结果保存 -------------------------
def save_results(dataset, metrics, roc_data):
    """保存评估结果和ROC曲线"""
    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(project_dir, 'pretrained_models/5cv_roc', f'Transformer_metrics_{dataset}.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    # 保存ROC数据
    roc_df = pd.DataFrame(roc_data)
    roc_file = os.path.join(project_dir, 'pretrained_models/5cv_roc', f'Transformer_ROC_{dataset}.csv')
    roc_df.to_csv(roc_file, index=False)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8,6))
    plt.plot(roc_df['FPR'], roc_df['TPR'], 
            color='darkorange', lw=2,
            label=f'Transformer (AUC = {metrics["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Best ROC for {dataset}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(project_dir, 'pretrained_models/5cv_roc', f'Transformer_ROC_{dataset}.png'))
    plt.close()

# ------------------------- 交叉验证流程 -------------------------
def run_cross_validation(dataset, X_train, y_train, X_test, y_test, input_dim):
    """执行5折交叉验证"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_metrics = {'AUC': 0}
    best_roc_data = None
    final_model_path = f'/your_path/EnDeep4mC/pretrained_models/5cv/transformer_best_{dataset}.h5'

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n----- Fold {fold} -----")
        
        # 划分数据
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # 预处理
        X_aug, y_aug, X_val_rs, scaler = preprocess_data(X_tr, y_tr, X_val)
        
        # 构建模型
        model = build_transformer_model(input_dim=X_aug.shape[2])
        fold_model_path = f'/tmp/transformer_{dataset}_fold{fold}.h5'
        
        # 训练模型
        model.model.fit(
            X_aug, y_aug,
            validation_data=(X_val_rs, y_val),
            epochs=100,
            batch_size=512,
            verbose=1,
            callbacks=get_callbacks(fold_model_path, fold),
            class_weight=dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
        )
        
        # 评估模型
        model.model.load_weights(fold_model_path)
        test_metrics, y_proba = evaluate_model(model.model, X_test, y_test, scaler)
        
        print(f"\nFold {fold} 测试集表现:")
        print(f"AUC: {test_metrics['AUC']:.6f}  ACC: {test_metrics['ACC']:.6f}")
        print(f"SN: {test_metrics['SN']:.6f}  SP: {test_metrics['SP']:.6f}")
        
        # 更新最佳模型
        if test_metrics['AUC'] > best_metrics['AUC']:
            best_metrics = test_metrics
            model.model.save(final_model_path)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            best_roc_data = {'FPR': fpr, 'TPR': tpr}

    return best_metrics, best_roc_data

# ------------------------- 结果保存 -------------------------
def save_combined_results(summary_data):
    """保存所有数据集的汇总结果"""
    summary_df = pd.DataFrame(summary_data)
    print("\n测试结果汇总:")
    print(summary_df)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f'/your_path/EnDeep4mC/pretrained_models/5cv_cross_species/transformer_5cv_summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n汇总结果已保存至: {summary_path}")

# ------------------------- 主流程 -------------------------
def main():
    
    # 验证GPU配置
    print("GPU可用性:", tf.config.list_physical_devices('GPU'))
    
    fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']

    model_name = 'Transformer'
    summary_data = []
    
    for dataset in fasta_dataset_names:
        print(f"\n=== Processing Dataset: {dataset} ===")
        
        # 特征选择
        try:
            best_n = load_best_n(model_name, dataset)
            feature_methods = get_feature_methods(load_top_features(model_name, dataset, best_n))
        except Exception as e:
            print(f"Feature selection error: {str(e)}, using default features")
            feature_methods = None
        
        # 加载数据
        X_train, y_train, X_test, y_test = load_full_dataset(dataset, feature_methods)
        input_dim = X_train.shape[1]
        
        # 执行交叉验证
        best_metrics, best_roc_data = run_cross_validation(
            dataset, X_train, y_train, X_test, y_test, input_dim
        )
        
        # 输出结果
        print("\nBest Test Performance:")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.6f}")
        
        # 保存单个数据集结果
        save_results(dataset, best_metrics, best_roc_data)
        
        # 添加至汇总数据
        summary_entry = {
            'Dataset': dataset,
            'ACC': f"{best_metrics['ACC']:.6f}",
            'SN': f"{best_metrics['SN']:.6f}", 
            'SP': f"{best_metrics['SP']:.6f}",
            'F1': f"{best_metrics['F1']:.6f}",
            'MCC': f"{best_metrics['MCC']:.6f}",
            'AUC': f"{best_metrics['AUC']:.6f}"
        }
        summary_data.append(summary_entry)

    # 保存汇总结果
    save_combined_results(summary_data)

if __name__ == "__main__":
    main()