import numpy as np
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, matthews_corrcoef, roc_curve)
from lightgbm import LGBMClassifier
import xgboost as xgb
from tqdm import tqdm
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
sns.set(style="whitegrid")

from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection import load_top_features, get_feature_methods


class BaseEnsemble:
    def __init__(self, mode='indiv', ablation_models=None):
        self.base_models = ['CNN', 'BLSTM', 'Transformer'] if not ablation_models else ablation_models
        self.mode = mode  # 'indiv' or '5cv'
        self.scalers = {}
        self.meta_model = None
        self.dataset_name = None
        self.feature_configs = {}
        self.full_data_cache = {}
        self.results = []
        self.visualization_data = []

        # path config
        self.model_dirs = {
            'indiv': {
                'pretrain': 'train_data_auto_indiv',
                'output': 'ablation_indiv_results'
            },
            '5cv': {
                'pretrain': 'train_data_auto_5cv',
                'output': 'ablation_5cv_results'
            }
        }

    def _get_model_path(self, model_name):
        """get the path of pretrained model (corrected file name format)"""
        suffix = 'h5' if model_name in ['CNN', 'BLSTM', 'Transformer'] else 'pkl'
        # adjust the file name according to the mode
        if self.mode == '5cv':
            filename = f"{model_name.lower()}_best_{self.dataset_name}.{suffix}"
        else:
            filename = f"{model_name.lower()}_{self.dataset_name}.{suffix}"
        return os.path.join(
            project_dir,
            self.model_dirs[self.mode]['pretrain'],
            filename
        )

    def _get_best_n_features(self, model_name):
        """get the optimal number of features"""
        acc_table_path = os.path.join(project_dir, "dl_feature_ranking", f"{model_name}_Feature_Acc_Table.csv")
        acc_table = pd.read_csv(acc_table_path)
        return int(acc_table[acc_table['N'] == 'best_n'][self.dataset_name].values[0])

    def _load_full_dataset(self, model_name):
        """load datasets and perform feature engineering"""
        if (model_name, self.dataset_name) in self.full_data_cache:
            return self.full_data_cache[(model_name, self.dataset_name)]

        base_dir = os.path.join(project_dir, 'data/4mC')
        paths = {
            'train_pos': os.path.join(base_dir, self.dataset_name, "train_pos.txt"),
            'train_neg': os.path.join(base_dir, self.dataset_name, "train_neg.txt"),
            'test_pos': os.path.join(base_dir, self.dataset_name, "test_pos.txt"),
            'test_neg': os.path.join(base_dir, self.dataset_name, "test_neg.txt")
        }

        # merge all data
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

        # feature engineering
        best_n = self._get_best_n_features(model_name)
        self.feature_configs[model_name] = get_feature_methods(
            load_top_features(model_name, self.dataset_name, best_n)
        )
        X_full, y_full, _ = ml_code(full_df, "training", self.feature_configs[model_name])

        # cache processing
        self.full_data_cache[(model_name, self.dataset_name)] = (X_full, y_full)
        return X_full, y_full

    def _prepare_data(self, model_name):
        """data preprocessing process"""
        X_full, y_full = self._load_full_dataset(model_name)
        
        # divide the training sets
        base_dir = os.path.join(project_dir, 'data/4mC')
        train_size = (
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_pos.txt"))) +
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_neg.txt")))
        )
        
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        X_test = X_full[train_size:]
        y_test = y_full[train_size:]

        # data enhancement
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        
        # standardization
        self.scalers[model_name] = StandardScaler()
        X_res_scaled = self.scalers[model_name].fit_transform(X_res)
        
        # add noise
        np.random.seed(42)
        noise = 0.1 * np.random.normal(0, 1, X_res_scaled.shape)
        X_train_aug = np.vstack([X_res_scaled, X_res_scaled + noise])
        y_train_aug = np.concatenate([y_res, y_res])
        
        X_test_scaled = self.scalers[model_name].transform(X_test)
        
        return X_train_aug, y_train_aug, X_test_scaled, y_test

    def _generate_meta_features(self, model_name, X):
        """generate meta features"""
        if model_name in ['CNN', 'BLSTM', 'Transformer']:
            X_reshaped = X.reshape(-1, 1, X.shape[1])
            model = tf.keras.models.load_model(self._get_model_path(model_name))
            return model.predict(X_reshaped, verbose=0).flatten()
        else:
            return X  # 对于其他模型直接返回原始特征

    def _get_meta_dataset(self, ablation_combination):
        """building a Meta Feature Dataset"""
        meta_features = []
        y_all = None
        
        for model_name in ablation_combination:
            if self.mode == 'indiv':
                X_train, y_train, X_test, y_test = self._prepare_data(model_name)
                train_feat = self._generate_meta_features(model_name, X_train)
                test_feat = self._generate_meta_features(model_name, X_test)
                meta_features.append(np.concatenate([train_feat, test_feat]))
                y_all = np.concatenate([y_train, y_test])
            else:
                X_all, y_all_model = self._prepare_base_data(model_name)
                meta_feat = self._generate_meta_features(model_name, X_all)
                meta_features.append(meta_feat)
                y_all = y_all_model

        return np.column_stack(meta_features), y_all

    def _evaluate_model(self, y_true, y_pred, y_proba):
        """model evaluation"""
        return {
            'ACC': accuracy_score(y_true, y_pred),
            'SN': recall_score(y_true, y_pred),
            'SP': recall_score(y_true, y_pred, pos_label=0),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_proba)
        }

    def _visualize_results(self, dataset):
        """viz results"""
        df = pd.DataFrame(self.visualization_data)
        plt.figure(figsize=(12, 8))
        
        # visualize relative changes
        baseline = df[df['combination'] == 'full'].iloc[0]
        df['delta'] = df.apply(lambda x: [(x[metric] - baseline[metric])/baseline[metric] 
                                        for metric in ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']], axis=1)
        
        metrics = ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']
        n_metrics = len(metrics)
        
        plt.figure(figsize=(15, 8))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            for comb in df['combination'].unique():
                if comb == 'full': continue
                values = df[df['combination'] == comb]['delta'].values[0]
                plt.bar(comb, values[i-1], 
                       label=f"{comb} ({'↑' if values[i-1]>0 else '↓'}{abs(values[i-1]*100):.1f}%)")
            
            plt.axhline(0, color='black', linestyle='--')
            plt.title(f"{metric} Relative Change")
            plt.xticks(rotation=45)
            if i == 1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_path = os.path.join(
            project_dir, 
            self.model_dirs[self.mode]['output'],
            f'ablation_{dataset}_visualization.png'
        )
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

class IndividualEnsemble(BaseEnsemble):
    """independent testing version"""
    def __init__(self, ablation_models=None):
        super().__init__(mode='indiv', ablation_models=ablation_models)
        
    def run_ablation(self, datasets):
        """ablation experiment"""
        for dataset in datasets:
            print(f"\n=== Processing {dataset} ===")
            self.dataset_name = dataset
            self.visualization_data = []
            
            # generate all possible combinations
            all_combinations = [['CNN', 'BLSTM', 'Transformer']]  # 完整模型
            for i in range(len(self.base_models)):
                all_combinations.append([m for j, m in enumerate(self.base_models) if j != i])
            
            for combination in all_combinations:
                comb_name = 'full' if len(combination)==3 else '-'.join([m for m in self.base_models if m not in combination])
                print(f"\nRunning ablation: {comb_name}")
                
                # prepare the meta data
                meta_X, y = self._get_meta_dataset(combination)
                X_train, X_test, y_train, y_test = train_test_split(
                    meta_X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # train the meta learner
                base_models = [
                    ('xgb', xgb.XGBClassifier(n_estimators=500)),
                    ('lgbm', LGBMClassifier(n_estimators=300))
                ]
                final_model = LogisticRegression(class_weight='balanced', max_iter=2000)
                
                stacker = StackingClassifier(
                    estimators=base_models,
                    final_estimator=final_model,
                    stack_method='predict_proba',
                    passthrough=True,
                    n_jobs=-1
                )
                stacker.fit(X_train, y_train)
                
                # evaluation
                y_proba = stacker.predict_proba(X_test)[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
                metrics = self._evaluate_model(y_test, y_pred, y_proba)

                self.visualization_data.append({
                    'combination': comb_name,
                    **metrics
                })
                
            # save and viz
            result_df = pd.DataFrame(self.visualization_data)
            output_dir = os.path.join(project_dir, self.model_dirs[self.mode]['output'])
            os.makedirs(output_dir, exist_ok=True)
            result_path = os.path.join(output_dir, f'ablation_{dataset}_results.csv')
            result_df.to_csv(result_path, index=False)
            
            self._visualize_results(dataset)

class CrossValEnsemble(BaseEnsemble):
    """cross validation version"""
    def __init__(self, ablation_models=None):
        super().__init__(mode='5cv', ablation_models=ablation_models)
        self.n_folds = 5

    def _prepare_base_data(self, model_name):
        """prepare cross validation data"""
        X_full, y_full = self._load_full_dataset(model_name)
        train_size = len(read_fasta_data(os.path.join(project_dir, 'data/4mC', self.dataset_name, "train_pos.txt"))) + \
                    len(read_fasta_data(os.path.join(project_dir, 'data/4mC', self.dataset_name, "train_neg.txt")))
        
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        X_test = X_full[train_size:]
        y_test = y_full[train_size:]
        
        # data enhancement
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        
        # standardization
        self.scalers[model_name] = StandardScaler()
        X_res_scaled = self.scalers[model_name].fit_transform(X_res)
        
        # merge the datasets
        X_all = np.vstack([X_res_scaled, self.scalers[model_name].transform(X_test)])
        y_all = np.concatenate([y_res, y_test])
        
        return X_all, y_all

    def run_ablation(self, datasets):
        """perform cross validation ablation experiments"""
        for dataset in datasets:
            print(f"\n=== Processing {dataset} ===")
            self.dataset_name = dataset
            self.visualization_data = []

            all_combinations = [['CNN', 'BLSTM', 'Transformer']]
            for i in range(len(self.base_models)):
                all_combinations.append([m for j, m in enumerate(self.base_models) if j != i])
            
            for combination in all_combinations:
                comb_name = 'full' if len(combination)==3 else '-'.join([m for m in self.base_models if m not in combination])
                print(f"\nRunning ablation: {comb_name}")

                meta_X, y = self._get_meta_dataset(combination)

                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
                fold_metrics = []
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(meta_X, y), 1):
                    X_train, X_test = meta_X[train_idx], meta_X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    base_models = [
                        ('xgb', xgb.XGBClassifier(n_estimators=500)),
                        ('lgbm', LGBMClassifier(n_estimators=300))
                    ]
                    final_model = LogisticRegression(class_weight='balanced', max_iter=2000)
                    
                    stacker = StackingClassifier(
                        estimators=base_models,
                        final_estimator=final_model,
                        stack_method='predict_proba',
                        passthrough=True,
                        n_jobs=-1
                    )
                    stacker.fit(X_train, y_train)

                    y_proba = stacker.predict_proba(X_test)[:, 1]
                    y_pred = (y_proba > 0.5).astype(int)
                    metrics = self._evaluate_model(y_test, y_pred, y_proba)
                    fold_metrics.append(metrics)

                avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
                self.visualization_data.append({
                    'combination': comb_name,
                    **avg_metrics
                })

            result_df = pd.DataFrame(self.visualization_data)
            output_dir = os.path.join(project_dir, self.model_dirs[self.mode]['output'])
            os.makedirs(output_dir, exist_ok=True)
            result_path = os.path.join(output_dir, f'ablation_{dataset}_results.csv')
            result_df.to_csv(result_path, index=False)
            
            self._visualize_results(dataset)

if __name__ == "__main__":
    # set configs
    datasets = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
    #datasets = ['4mC_G.subterraneus', '4mC_G.pickeringii']
    modes = ['indiv', '5cv']  # 两种运行模式
    #modes = ['5cv']
    
    for mode in modes:
        print(f"\n=== Running in {mode.upper()} mode ===")
        
        if mode == 'indiv':
            ensemble = IndividualEnsemble()
        else:
            ensemble = CrossValEnsemble()
        
        ensemble.run_ablation(datasets)