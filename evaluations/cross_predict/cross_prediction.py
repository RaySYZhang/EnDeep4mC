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
import time

# ================== Configs ==================
# Species classification
# There are some datasets split from the original benchmark datasets of the same name, so a ‘2’ is added to distinguish them.
SPECIES_CATEGORIES = {
    'Plant': {
        'color': 'green',
        'datasets': ['4mC_A.thaliana2', '4mC_C.equisetifolia', '4mC_F.vesca', '4mC_R.chinensis']
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
MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models/5cv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'evaluations/cross_predict_12')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(PROJECT_DIR)
from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection_eukaryotes import load_top_features, get_feature_methods

# ================== Universal function ==================
def load_best_n(model_name, dataset):
    """Load the optimal number of features"""
    print(f"🔄 加载最优特征数量: {model_name} - {dataset}")
    acc_table_path = os.path.join(PROJECT_DIR, "feature_engineering/ifs_result_cross_species", 
                                 f"{model_name}_Feature_Acc_Table.csv")
    
    # 检查文件是否存在
    if not os.path.exists(acc_table_path):
        print(f"❌ 特征表文件不存在: {acc_table_path}")
        return None
    
    df = pd.read_csv(acc_table_path)
    
    # 检查数据集列是否存在
    if dataset not in df.columns:
        print(f"❌ 数据集 '{dataset}' 不在特征表中")
        return None
    
    best_n = df[df['N'] == 'best_n'][dataset].values[0]
    return int(float(best_n))

def prepare_source_data(source_dataset, model_type):
    """Prepare training data for the source species"""
    print(f"🔄 准备源数据: {source_dataset} ({model_type})")
    
    # Load training data
    train_pos = os.path.join(DATA_DIR, source_dataset, "train_pos.txt")
    train_neg = os.path.join(DATA_DIR, source_dataset, "train_neg.txt")
    
    # 添加文件检查
    if not os.path.exists(train_pos):
        print(f"❌ 文件不存在: {train_pos}")
        return None, None, None
    if not os.path.exists(train_neg):
        print(f"❌ 文件不存在: {train_neg}")
        return None, None, None
    
    # 读取数据
    try:
        pos_data = read_fasta_data(train_pos)
        neg_data = read_fasta_data(train_neg)
    except Exception as e:
        print(f"❌ 读取数据时出错: {str(e)}")
        return None, None, None
    
    # create dataFrame
    df = pd.DataFrame({
        "label": [1]*len(pos_data) + [0]*len(neg_data),
        "seq": pos_data + neg_data
    })
    
    # Obtain feature configuration
    best_n = load_best_n(model_type, source_dataset)
    if best_n is None:
        print(f"❌ 无法获取最优特征数量")
        return None, None, None
    
    features = load_top_features(model_type, source_dataset, best_n)
    feature_methods = get_feature_methods(features)
    
    # generate features
    try:
        X_train, y_train, _ = ml_code(df, "training", feature_methods)
    except Exception as e:
        print(f"❌ 生成特征时出错: {str(e)}")
        return None, None, None
    
    # 添加完成提示
    print(f"✅ 源数据准备完成: {source_dataset} ({model_type}) - 样本数: {len(X_train)}")
    return X_train, y_train, feature_methods

def prepare_target_data(target_dataset, feature_methods):
    """Prepare target test data"""
    print(f"🔄 准备目标数据: {target_dataset}")
    
    test_pos = os.path.join(DATA_DIR, target_dataset, "test_pos.txt")
    test_neg = os.path.join(DATA_DIR, target_dataset, "test_neg.txt")
    
    # 添加文件检查
    if not os.path.exists(test_pos):
        print(f"❌ 文件不存在: {test_pos}")
        return None, None
    if not os.path.exists(test_neg):
        print(f"❌ 文件不存在: {test_neg}")
        return None, None
    
    # 读取数据
    try:
        pos_data = read_fasta_data(test_pos)
        neg_data = read_fasta_data(test_neg)
    except Exception as e:
        print(f"❌ 读取数据时出错: {str(e)}")
        return None, None
    
    df = pd.DataFrame({
        "label": [1]*len(pos_data) + [0]*len(neg_data),
        "seq": pos_data + neg_data
    })
    
    try:
        X_test, y_test, _ = ml_code(df, "testing", feature_methods)
    except Exception as e:
        print(f"❌ 生成特征时出错: {str(e)}")
        return None, None
    
    # 添加完成提示
    print(f"✅ 目标数据准备完成: {target_dataset} - 样本数: {len(X_test)}")
    return X_test, y_test

def predict_single_case(source, target):
    """Perform single cross prediction"""
    try:
        start_time = time.time()  # 记录开始时间
        
        # 添加进度提示
        print(f"\n🔁 开始处理: {source} -> {target}")
        
        # Load ensemble model
        ensemble_path = os.path.join(MODEL_DIR, f'ensemble_5cv_{source}.pkl')
        print(f"🔄 加载集成模型: {ensemble_path}")
        
        # 检查集成模型文件是否存在
        if not os.path.exists(ensemble_path):
            print(f"❌ 集成模型文件不存在: {ensemble_path}")
            return np.nan, np.nan
        
        try:
            ensemble_model = joblib.load(ensemble_path)
        except Exception as e:
            print(f"❌ 加载集成模型时出错: {str(e)}")
            return np.nan, np.nan
        
        meta_features = []
        y_target = None
        
        for model_type in BASE_MODELS:
            model_start = time.time()  # 记录模型开始时间
            print(f"  🔄 处理基础模型: {model_type}")
            
            # 准备源数据
            X_source_train, _, feature_methods = prepare_source_data(source, model_type)
            if X_source_train is None:
                print(f"  ❌ 源数据准备失败")
                return np.nan, np.nan
                
            # 准备目标数据
            X_target, y_target = prepare_target_data(target, feature_methods)
            if X_target is None:
                print(f"  ❌ 目标数据准备失败")
                return np.nan, np.nan
                
            # 标准化
            print(f"  🔄 标准化数据")
            try:
                scaler = StandardScaler().fit(X_source_train)
                X_target_scaled = scaler.transform(X_target)
            except Exception as e:
                print(f"  ❌ 标准化失败: {str(e)}")
                return np.nan, np.nan
            
            # 加载基础模型
            model_path = os.path.join(MODEL_DIR, f'{model_type.lower()}_best_{source}.h5')
            print(f"  🔄 加载模型: {model_path}")
            
            # 添加模型文件检查
            if not os.path.exists(model_path):
                print(f"  ❌ 模型文件不存在: {model_path}")
                return np.nan, np.nan
                
            try:
                base_model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"  ❌ 加载模型时出错: {str(e)}")
                return np.nan, np.nan
            
            # 预测
            print(f"  🔄 进行预测")
            try:
                input_data = X_target_scaled.reshape(-1, 1, X_target_scaled.shape[1])
                preds = base_model.predict(input_data, verbose=0).flatten()
                meta_features.append(preds)
            except Exception as e:
                print(f"  ❌ 预测失败: {str(e)}")
                return np.nan, np.nan
            
            # 添加模型处理完成提示
            model_time = time.time() - model_start
            print(f"  ✅ {model_type} 模型处理完成 - 耗时: {model_time:.2f}秒")
        
        # Composite elemental features
        meta_X = np.column_stack(meta_features)
        
        # Integrated prediction
        print(f"🔄 进行集成预测")
        try:
            y_pred = ensemble_model.predict(meta_X)
            if hasattr(ensemble_model, 'predict_proba'):
                y_proba = ensemble_model.predict_proba(meta_X)[:, 1]
            else:
                y_proba = y_pred
        except Exception as e:
            print(f"❌ 集成预测失败: {str(e)}")
            return np.nan, np.nan
        
        # 计算指标
        try:
            acc = accuracy_score(y_target, y_pred)
            auc = roc_auc_score(y_target, y_proba)
        except Exception as e:
            print(f"❌ 计算指标失败: {str(e)}")
            return np.nan, np.nan
        
        # 添加完成提示
        total_time = time.time() - start_time
        print(f"✅ 完成处理: {source} -> {target} - 准确率: {acc:.4f}, AUC: {auc:.4f} - 总耗时: {total_time:.2f}秒")
        
        return acc, auc
    
    except Exception as e:
        print(f"❌ 处理 {source}->{target} 时出错: {str(e)}")
        return np.nan, np.nan

# ================== Visualization ==================
def generate_heatmap(matrix, metric_name):
    print(f"🖼️ 生成{metric_name}热力图")
    
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
    
    print(f"✅ {metric_name}热力图已保存: {png_path}")
    print(f"✅ {metric_name}矩阵已保存: {csv_path}")

# ================== Main ==================
def main():
    # 添加总体开始提示
    print(f"🚀 开始跨数据集预测任务")
    print(f"数据集数量: {len(ORDERED_DATASETS)}")
    print(f"总任务数: {len(ORDERED_DATASETS)**2}")
    print(f"使用线程数: {os.cpu_count()}")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
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
        progress = tqdm(total=len(futures), desc="处理跨数据集预测")
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

    # 添加可视化提示
    print(f"🖼️ 生成准确率热力图")
    generate_heatmap(acc_matrix, 'Accuracy')
    
    print(f"🖼️ 生成AUC热力图")
    generate_heatmap(auc_matrix, 'AUC')
    
    # 添加总体完成提示
    print(f"🎉 任务完成! 结果已保存至: {OUTPUT_DIR}")
    
    # 计算并显示NaN值的数量
    nan_count_acc = acc_matrix.isna().sum().sum()
    nan_count_auc = auc_matrix.isna().sum().sum()
    print(f"准确率矩阵中缺失值数量: {nan_count_acc}")
    print(f"AUC矩阵中缺失值数量: {nan_count_auc}")

if __name__ == "__main__":
    main()