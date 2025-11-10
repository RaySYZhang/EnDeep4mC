import numpy as np
import datetime
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_curve)
from lightgbm import LGBMClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import warnings
import time

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
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n, get_feature_names

class EnhancedEnsemble5CV:
    def __init__(self, n_folds=5):
        self.base_models = ['CNN', 'BLSTM', 'Transformer']
        self.scalers = {}
        self.meta_model = None
        self.dataset_name = None
        self.n_folds = n_folds
        self.feature_configs = {}
        self.best_meta_model = None
        self.full_data_cache = {}
        self.fold_metrics = []

    def _get_best_n_features(self, model_name):
        try:
            return load_best_n(model_name, self.dataset_name)
        except Exception as e:
            print(f"加载最佳特征数失败: {str(e)}, 使用默认值10")
            return 10

    def _load_full_dataset(self):
        if self.dataset_name in self.full_data_cache:
            return self.full_data_cache[self.dataset_name]
        
        base_dir = '/your_path/EnDeep4mC/data/4mC'
        paths = {
            'train_pos': os.path.join(base_dir, self.dataset_name, "train_pos.txt"),
            'train_neg': os.path.join(base_dir, self.dataset_name, "train_neg.txt"),
            'test_pos': os.path.join(base_dir, self.dataset_name, "test_pos.txt"),
            'test_neg': os.path.join(base_dir, self.dataset_name, "test_neg.txt")
        }

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
        
        self.full_data_cache[self.dataset_name] = full_df
        return full_df

    def _prepare_base_data(self, model_name):
        base_dir = '/your_path/EnDeep4mC/data/4mC'
        full_df = self._load_full_dataset()
        best_n = self._get_best_n_features(model_name)
        selected_features = load_top_features(model_name, self.dataset_name, best_n)
        self.feature_configs[model_name] = get_feature_methods(selected_features)

        # 获取完整特征信息
        X_full, y_full, feature_names = ml_code(full_df, "training", self.feature_configs[model_name])
        
        # 生成特征索引和名称
        feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
        feature_list = get_feature_names(selected_features)
        
        # 保存特征配置
        config_dir = os.path.join(project_dir, 'pretrained_models', '5cv', 'feature_configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f'{model_name}_{self.dataset_name}_config.pkl')
        joblib.dump({
            'selected_features': feature_list,
            'feature_methods': self.feature_configs[model_name],
            'feature_indices': feature_indices
        }, config_path)

        train_size = len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_pos.txt"))) + \
                    len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_neg.txt")))
        
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        X_test = X_full[train_size:]
        y_test = y_full[train_size:]

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        
        # 标准化处理及保存
        self.scalers[model_name] = StandardScaler()
        X_res_scaled = self.scalers[model_name].fit_transform(X_res)
        
        # 保存标准化器（新增部分）
        scaler_dir = os.path.join(project_dir, 'pretrained_models', '5cv', 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, f'{model_name}_{self.dataset_name}_scaler.pkl')
        joblib.dump(self.scalers[model_name], scaler_path)
        
        # 添加噪声增强
        np.random.seed(42)
        noise = 0.1 * np.random.normal(0, 1, X_res_scaled.shape)
        X_train_aug = np.vstack([X_res_scaled, X_res_scaled + noise])
        y_train_aug = np.concatenate([y_res, y_res])
        
        X_test_scaled = self.scalers[model_name].transform(X_test)
        
        X_all = np.vstack([X_train_aug, X_test_scaled])
        y_all = np.concatenate([y_train_aug, y_test])
        
        return X_all, y_all

    def _load_pretrained_model(self, model_name):
        model_path = f'/your_path/EnDeep4mC/pretrained_models/5cv/{model_name.lower()}_best_{self.dataset_name}.h5'
        return tf.keras.models.load_model(model_path)

    def _generate_meta_features(self, model_name, X):
        X_reshaped = X.reshape(-1, 1, X.shape[1])
        return self._load_pretrained_model(model_name).predict(X_reshaped, verbose=0).flatten()

    def _prepare_meta_dataset(self):
        meta_features = []
        y_all = None
        
        for model_name in tqdm(self.base_models, desc="基模型特征生成"):
            X_all, y_all_model = self._prepare_base_data(model_name)
            meta_feature = self._generate_meta_features(model_name, X_all)
            meta_features.append(meta_feature)
            
            if y_all is None:
                y_all = y_all_model
            else:
                assert np.array_equal(y_all, y_all_model), "标签不一致"

        return np.column_stack(meta_features), y_all

    def _log_metrics(self, y_true, y_pred, y_proba, stage="val"):
        return {
            f'{stage}_acc': accuracy_score(y_true, y_pred),
            f'{stage}_sn': recall_score(y_true, y_pred),
            f'{stage}_sp': recall_score(y_true, y_pred, pos_label=0),
            f'{stage}_f1': f1_score(y_true, y_pred),
            f'{stage}_mcc': matthews_corrcoef(y_true, y_pred),
            f'{stage}_auc': roc_auc_score(y_true, y_proba)
        }

    def fit_and_validate(self):
        meta_X, y = self._prepare_meta_dataset()
        
        base_models = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=20,
                verbose=-1
            ))
        ]

        final_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=0.6,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        best_auc = 0
        self.fold_metrics = []

        with tqdm(total=self.n_folds, desc="整体进度", position=0) as pbar:
            for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y), 1):
                start_time = time.time()
                
                X_train, X_val = meta_X[train_idx], meta_X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                class_weights = {
                    0: len(y_train)/(2*np.bincount(y_train)[0]),
                    1: len(y_train)/(2*np.bincount(y_train)[1])
                }
                
                stacker = StackingClassifier(
                    estimators=base_models,
                    final_estimator=clone(final_model).set_params(class_weight=class_weights),
                    stack_method='predict_proba',
                    passthrough=True,
                    n_jobs=-1
                )
                
                # 训练带进度条
                with tqdm(total=100, desc=f"Fold {fold} 训练进度", leave=False) as inner_pbar:
                    stacker.fit(X_train, y_train)
                    inner_pbar.update(100)
                
                # 训练集评估
                train_proba = stacker.predict_proba(X_train)[:, 1]
                train_pred = (train_proba > 0.5).astype(int)
                train_metrics = self._log_metrics(y_train, train_pred, train_proba, "train")
                
                # 验证集评估
                val_proba = stacker.predict_proba(X_val)[:, 1]
                val_pred = (val_proba > 0.5).astype(int)
                val_metrics = self._log_metrics(y_val, val_pred, val_proba, "val")
                
                # 记录指标
                fold_metric = {
                    'fold': fold,
                    **train_metrics,
                    **val_metrics,
                    'time': time.time() - start_time
                }
                self.fold_metrics.append(fold_metric)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '最佳AUC': best_auc,
                    '当前AUC': fold_metric['val_auc'],
                    '训练ACC': fold_metric['train_acc']
                })
                
                # 打印折叠结果
                print(f"\nFold {fold} 结果:")
                print(f"训练集 ACC: {fold_metric['train_acc']:.4f}  AUC: {fold_metric['train_auc']:.4f}")
                print(f"验证集 ACC: {fold_metric['val_acc']:.4f}  AUC: {fold_metric['val_auc']:.4f}")
                print(f"耗时: {fold_metric['time']:.1f}s")

                if fold_metric['val_auc'] > best_auc:
                    best_auc = fold_metric['val_auc']
                    self.best_meta_model = stacker
                    print(f"发现更佳元模型, AUC提升至: {best_auc:.4f}")

        # 保存模型和结果
        model_path = f'/your_path/EnDeep4mC/pretrained_models/5cv/ensemble_5cv_{self.dataset_name}.pkl'
        joblib.dump(self.best_meta_model, model_path)
        print(f"\n模型已保存至: {model_path}")
        
        # 保存ROC曲线
        val_proba = self.best_meta_model.predict_proba(meta_X)[:, 1]
        fpr, tpr, _ = roc_curve(y, val_proba)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_path = f'/your_path/EnDeep4mC/pretrained_models/5cv_roc/Ensemble_ROC_{self.dataset_name}'
        roc_df.to_csv(f"{roc_path}.csv", index=False)
        
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble (AUC = {best_auc:.4f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Ensemble ROC Curve ({self.dataset_name})')
        plt.legend(loc="lower right")
        plt.savefig(f"{roc_path}.png")
        plt.close()

        return pd.DataFrame(self.fold_metrics)

if __name__ == "__main__":
    
    datasets = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']

    final_results = []
    
    for dataset in datasets:
        print(f"\n=== 处理数据集: {dataset} ===")
        try:
            ensemble = EnhancedEnsemble5CV(n_folds=5)
            ensemble.dataset_name = dataset
            
            fold_results = ensemble.fit_and_validate()
            fold_results.to_csv(
                f'/your_path/EnDeep4mC/pretrained_models/5cv/fold_metrics_{dataset}.csv',
                index=False,
                float_format='%.6f'
            )
            
            # 汇总结果
            avg_metrics = {
                'Dataset': dataset,
                'ACC': fold_results['val_acc'].mean(),
                'SN': fold_results['val_sn'].mean(),
                'SP': fold_results['val_sp'].mean(),
                'F1': fold_results['val_f1'].mean(),
                'MCC': fold_results['val_mcc'].mean(),
                'AUC': fold_results['val_auc'].mean()
            }
            final_results.append(avg_metrics)
            
            print(f"\n{dataset} 平均性能:")
            for k, v in avg_metrics.items():
                if k != 'Dataset':
                    print(f"{k}: {v:.6f}")

        except Exception as e:
            print(f"处理 {dataset} 时出错: {str(e)}")
            continue
    
    # 保存最终结果
    result_df = pd.DataFrame(final_results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'/your_path/EnDeep4mC/pretrained_models/5cv/ensemble_5cv_summary_{timestamp}.csv'
    result_df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"\n最终结果已保存至: {save_path}")