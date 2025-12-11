import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import read_fasta_data, ml_code
from feature_engineering.feature_selection import get_feature_methods

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file upload to 16MB

# Global Configs
SPECIES_LIST = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
                '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
MODELS_DIR = os.path.join(project_dir, 'pretrained_models', '5cv')

# Initialize thread/process pool
model_executor = ThreadPoolExecutor(max_workers=4)
feature_executor = ProcessPoolExecutor(max_workers=4)

# Model cache
model_cache = {}

class ModelLoader:
    """Enhanced model loader, supporting dynamic feature method reconstruction"""
    @staticmethod
    def load_ensemble(species):
        return joblib.load(os.path.join(MODELS_DIR, f'ensemble_5cv_{species}.pkl'))
    
    @staticmethod
    def load_dl_model(model_name, species):
        return tf.keras.models.load_model(
            os.path.join(MODELS_DIR, f'{model_name.lower()}_best_{species}.h5')
        )
    
    @staticmethod
    def load_config(model_name, species):
        config_path = os.path.join(MODELS_DIR, 'feature_configs', 
                                 f'{model_name}_{species}_config.pkl')
        try:
            config = joblib.load(config_path)
        except FileNotFoundError:
            raise RuntimeError(f"Feature configs file: {model_name}_{species} not Found")

        # Verify necessary fields
        required_fields = ['selected_features', 'feature_indices']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"The config file needs necessary fields:'{field}': {os.path.basename(config_path)}")

        # Dynamically Reconstruct features_methods
        if 'feature_methods' not in config:
            try:
                config['feature_methods'] = get_feature_methods(config['selected_features'])
                app.logger.info(f"Dynamic reconstruction feature method: {model_name}_{species}")
            except Exception as e:
                raise ValueError(f"Feature method reconstruction failed: {str(e)}")

        # Feature index compatibility processing
        if not config.get('feature_indices'):
            try:
                scaler = joblib.load(os.path.join(MODELS_DIR, 
                                f'{model_name}_{species}_scaler.pkl'))
                config['feature_indices'] = list(range(scaler.n_features_in_))
                app.logger.warning(f"Automatically generate feature index: {model_name}_{species}")
            except Exception as e:
                raise ValueError(f"Unable to generate feature index: {str(e)}")

        return config
    
    @staticmethod
    def load_scaler(model_name, species):
        return joblib.load(os.path.join(MODELS_DIR, 'scalers', f'{model_name}_{species}_scaler.pkl'))

def load_species_models(species):
    """Enhanced model loading, including feature configuration verification"""
    if species in model_cache:
        return model_cache[species]
    
    try:
        futures = {
            'models': model_executor.submit(ModelLoader.load_ensemble, species),
            'base_models': {m: model_executor.submit(ModelLoader.load_dl_model, m, species) 
                          for m in BASE_MODELS},
            'configs': {m: model_executor.submit(ModelLoader.load_config, m, species) 
                       for m in BASE_MODELS},
            'scalers': {m: model_executor.submit(ModelLoader.load_scaler, m, species) 
                       for m in BASE_MODELS}
        }
        
        loaded = {
            'models': futures['models'].result(),
            'base_models': {},
            'feature_configs': {},
            'scalers': {}
        }
        
        for model_name in BASE_MODELS:
            config = futures['configs'][model_name].result()
            
            # Final verification
            if not config['feature_indices']:
                raise ValueError(f"Empty feature index: {model_name}_{species}")
            if not config['feature_methods']:
                raise ValueError(f"Empty feature method: {model_name}_{species}")
                
            loaded['base_models'][model_name] = futures['base_models'][model_name].result()
            loaded['feature_configs'][model_name] = config
            loaded['scalers'][model_name] = futures['scalers'][model_name].result()

        model_cache[species] = loaded
        return loaded
    
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {str(e)}")

class SequenceProcessor:
    """Enhanced sequence processor with length standardization"""
    @staticmethod
    def _parse_fasta(content):
        sequences = []
        current_seq = []
        seq_id = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append((''.join(current_seq), seq_id))
                    current_seq = []
                seq_id = line[1:].strip()[:100]
            elif line:
                current_seq.append(line.upper())
        
        if current_seq:
            sequences.append((''.join(current_seq), seq_id))
        
        # 不去重，直接返回所有序列
        return sequences

    @staticmethod
    def _standardize_sequence_length(seq, seq_id):
        """Standardize sequence length to 41bp for model input"""
        L = len(seq)
        
        # Case 1: Exactly 41bp - use directly
        if L == 41:
            return [(seq, seq_id, 21, 0, seq_id, '41bp')]  # (sequence, window_id, center_pos, window_start, original_seq_id, length_type)
        
        # Case 2: Shorter than 41bp - pad with base repetition
        elif L < 41:
            # Calculate padding needed on each side
            total_pad = 41 - L
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
            
            # For left padding, repeat sequence enough times and take last left_pad characters
            left_padding = ""
            if left_pad > 0:
                repeat_count = (left_pad // L) + 2
                extended_seq = seq * repeat_count
                left_padding = extended_seq[-left_pad:]
            
            # For right padding, repeat sequence enough times and take first right_pad characters
            right_padding = ""
            if right_pad > 0:
                repeat_count = (right_pad // L) + 2
                extended_seq = seq * repeat_count
                right_padding = extended_seq[:right_pad]
            
            # Build padded sequence
            padded_seq = left_padding + seq + right_padding
            
            # Calculate center position in original sequence (1-based)
            center_pos_in_original = L // 2 + 1 if L % 2 == 1 else L // 2
            
            return [(padded_seq, f"{seq_id}_padded", center_pos_in_original, 0, seq_id, 'short')]
        
        # Case 3: Longer than 41bp - sliding window segmentation
        else:
            windows = []
            # Use sliding window with step size 1
            for i in range(L - 40):
                window = seq[i:i+41]
                center_base = window[20]  # 21st base (0-indexed 20)
                
                # Only keep windows where center is C (cytosine)
                if center_base == 'C':
                    windows.append((window, f"{seq_id}_pos{i+21}", i+21, i, seq_id, 'long_window'))
            
            # If no windows with C at center, use all windows (for compatibility)
            if not windows:
                for i in range(L - 40):
                    window = seq[i:i+41]
                    windows.append((window, f"{seq_id}_pos{i+21}", i+21, i, seq_id, 'long_window'))
            
            return windows

    def process_input(self, text_input, file_input):
        all_windows = []
        
        # Parse input sequences
        raw_sequences = []
        if text_input.strip():
            raw_sequences += self._parse_fasta(text_input)
        if file_input and file_input.filename:
            try:
                content = file_input.read().decode('utf-8')
                raw_sequences += self._parse_fasta(content)
            except UnicodeDecodeError:
                raise ValueError("File encoding error: Only UTF-8 encoding is supported")
        
        # Process each sequence through standardization pipeline
        for seq, seq_id in raw_sequences:
            if not self._validate_sequence(seq):
                raise ValueError(f"Invalid sequence characters in {seq_id}. Only A, T, C, G allowed.")
            
            windows = self._standardize_sequence_length(seq, seq_id)
            
            # Add each window to the list
            for window_seq, window_id, center_pos, window_start, original_seq_id, length_type in windows:
                all_windows.append({
                    'seq_id': window_id,
                    'seq': window_seq,
                    'label': 0,
                    'is_train': False,
                    'original_seq_id': original_seq_id,
                    'center_pos': center_pos,
                    'window_start': window_start,
                    'length_type': length_type,
                    'original_length': len(seq)
                })
        
        # Build DataFrame with window information
        if all_windows:
            return pd.DataFrame(all_windows)
        else:
            return pd.DataFrame()

    @staticmethod
    def _validate_sequence(seq):
        valid_chars = {'A', 'T', 'C', 'G'}
        return 20 <= len(seq) <= 100000 and all(c in valid_chars for c in seq)

class FeatureGenerator:
    """Optimized feature generator"""
    def __init__(self, feature_config, scaler):
        self.feature_methods = feature_config['feature_methods']
        self.feature_indices = np.array(feature_config['feature_indices'])
        self.scaler = scaler
        self.cache = {}
    
    def generate(self, seq_df):
        cached_features = []
        new_sequences = []
        
        # Cache processing
        for _, row in seq_df.iterrows():
            cache_key = f"{row['seq']}_{row['center_pos']}"
            if cache_key in self.cache:
                cached_features.append(self.cache[cache_key])
            else:
                new_sequences.append((row.to_dict(), cache_key))
        
        # Parallel processing of new data
        if new_sequences:
            gen_func = partial(self._generate_single, 
                             feature_methods=self.feature_methods,
                             feature_indices=self.feature_indices)
            
            results = []
            for row_data, cache_key in new_sequences:
                try:
                    feat = gen_func(row_data)
                    self.cache[cache_key] = feat
                    cached_features.append(feat)
                except Exception as e:
                    raise RuntimeError(f"Feature generation failed for sequence {row_data['seq_id']}: {str(e)}")
        
        features = np.vstack(cached_features) if cached_features else np.array([]).reshape(0, len(self.feature_indices))
        
        # Final standardization
        if features.size > 0 and features.shape[1] != self.scaler.n_features_in_:
            raise ValueError(f"Feature dimension mismatch: {self.scaler.n_features_in_} vs {features.shape[1]}")
        
        return self.scaler.transform(features) if features.size > 0 else features
    
    @staticmethod
    def _generate_single(row_data, feature_methods, feature_indices):
        """Thread safety feature generation"""
        df = pd.DataFrame([{
            'seq_id': row_data['seq_id'],
            'seq': row_data['seq'],
            'label': 0,
            'is_train': False
        }])
        
        try:
            X, _, _ = ml_code(df, "testing", feature_methods)
            return X[0][feature_indices]
        except Exception as e:
            raise RuntimeError(f"Feature generation failed: {row_data['seq'][:50]}... {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        species = request.form['species']
        text_input = request.form.get('sequence', '')
        file_input = request.files.get('file')
        
        if species not in SPECIES_LIST:
            return jsonify({'error': 'Invalid species selection'}), 400
        
        processor = SequenceProcessor()
        input_df = processor.process_input(text_input, file_input)
        
        if input_df.empty:
            return jsonify({'error': 'No effective DNA sequences were found or no windows with C at center for long sequences'}), 400
        
        # Log processing information
        original_seqs = input_df['original_seq_id'].unique()
        app.logger.info(f"Processed {len(original_seqs)} original sequences into {len(input_df)} windows")
        
        models = load_species_models(species)
        
        meta_features = []
        for model_name in BASE_MODELS:
            generator = FeatureGenerator(
                feature_config=models['feature_configs'][model_name],
                scaler=models['scalers'][model_name]
            )
            X = generator.generate(input_df)
            
            if X.size == 0:
                continue
                
            dl_model = models['base_models'][model_name]
            dl_input = X.reshape(-1, 1, X.shape[1])
            pred_proba = dl_model.predict(dl_input, verbose=0).flatten()
            meta_features.append(pred_proba)
        
        if not meta_features:
            return jsonify({'error': 'No valid features generated for prediction'}), 400
            
        meta_X = np.column_stack(meta_features)
        probabilities = models['models'].predict_proba(meta_X)[:, 1]
        
        # Group results by original sequence and process differently based on sequence type
        results_by_sequence = {}
        
        for idx, (_, row) in enumerate(input_df.iterrows()):
            original_seq_id = row['original_seq_id']
            length_type = row['length_type']
            seq_length = len(row['seq'])
            original_length = row['original_length']
            
            if original_seq_id not in results_by_sequence:
                results_by_sequence[original_seq_id] = {
                    'original_seq': row['seq'] if seq_length <= 100 else row['seq'][:100] + "...",
                    'length_type': length_type,
                    'original_length': original_length,
                    'is_long_sequence': original_length > 41,
                    'predictions': [],
                    'probabilities': [],
                    'window_details': []  # Store window details for long sequences
                }
            
            # Store prediction for this window
            window_prediction = {
                'window_id': row['seq_id'],
                'probability': float(round(probabilities[idx], 4)),
                'center_position': row['center_pos'],
                'window_sequence': row['seq'],
                'is_4mC_site': bool(probabilities[idx] >= 0.5)
            }
            
            results_by_sequence[original_seq_id]['predictions'].append(window_prediction)
            results_by_sequence[original_seq_id]['probabilities'].append(probabilities[idx])
            results_by_sequence[original_seq_id]['window_details'].append(window_prediction)
        
        # Process results for each original sequence
        final_results = []
        window_details_results = []
        
        for original_seq_id, seq_data in results_by_sequence.items():
            is_long_sequence = seq_data['is_long_sequence']
            predictions = seq_data['predictions']
            probabilities_list = seq_data['probabilities']
            
            if is_long_sequence and len(predictions) > 0:
                # For long sequences: calculate overall average probability
                overall_prob = np.mean(probabilities_list)
                
                # Count cytosine positions with 4mC site
                cytosine_4mC_count = sum(1 for p in predictions if p['is_4mC_site'])
                total_cytosine_positions = len(predictions)
                
                # Create main result for the whole long sequence
                final_results.append({
                    'seq_id': original_seq_id,
                    'original_seq_id': original_seq_id,
                    'sequence': seq_data['original_seq'],
                    'probability': float(round(overall_prob, 4)),
                    'is_4mC_site': bool(overall_prob >= 0.5),
                    'center_position': None,  # Not applicable for whole sequence
                    'is_aggregated': True,
                    'sequence_length': 'long',
                    'original_length': seq_data['original_length'],
                    'cytosine_positions': total_cytosine_positions,
                    '4mC_positions': cytosine_4mC_count,
                    'type': 'whole_sequence'
                })
                
                # Store window details separately
                for pred in predictions:
                    window_details_results.append({
                        'seq_id': pred['window_id'],
                        'original_seq_id': original_seq_id,
                        'sequence': pred['window_sequence'],
                        'probability': pred['probability'],
                        'is_4mC_site': pred['is_4mC_site'],
                        'center_position': pred['center_position'],
                        'is_aggregated': False,
                        'sequence_length': '41bp_window',
                        'type': 'window_detail'
                    })
                
            else:
                # For short sequences or single windows (including exactly 41bp)
                # Take the first prediction (there should be only one)
                if predictions:
                    pred = predictions[0]
                    final_results.append({
                        'seq_id': pred['window_id'],
                        'original_seq_id': original_seq_id,
                        'sequence': pred['window_sequence'],
                        'probability': pred['probability'],
                        'is_4mC_site': pred['is_4mC_site'],
                        'center_position': pred['center_position'],
                        'is_aggregated': False,
                        'sequence_length': '41bp' if seq_data['original_length'] == 41 else 'short',
                        'original_length': seq_data['original_length'],
                        'type': 'direct'
                    })
        
        # Add processing summary
        long_sequences = sum(1 for seq_id in results_by_sequence 
                           if results_by_sequence[seq_id]['is_long_sequence'])
        short_sequences = len(results_by_sequence) - long_sequences
        
        summary = {
            'total_original_sequences': len(original_seqs),
            'total_windows_processed': len(input_df),
            'total_final_predictions': len(final_results),
            'total_window_details': len(window_details_results),
            'long_sequences': long_sequences,
            'short_sequences': short_sequences,
            '41bp_sequences': sum(1 for r in final_results if r.get('sequence_length') == '41bp')
        }
        
        return jsonify({
            'final_results': final_results,
            'window_details': window_details_results,
            'summary': summary,
            'message': f"Processed {len(original_seqs)} sequence(s): {long_sequences} long, {short_sequences} short/41bp"
        })
    
    except ValueError as e:
        app.logger.error(f"Input validation failed: {str(e)}")
        return jsonify({'error': 'Input validation failed', 'detail': str(e)}), 400
    except KeyError as e:
        app.logger.error(f"Parameter missing: {str(e)}")
        return jsonify({'error': f'Parameter missing: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Abnormal prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Processing failed', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7012, threaded=True)  # 确保host是0.0.0.0
    # 运行服务：nohup python3 /home/zhangshuyu/EnDeep4mC/web_server/app_5cv.py > /home/zhangshuyu/EnDeep4mC/log/app.log 2>&1 &
    # 公网访问：http://112.124.26.17:7012/
    # 监控日志：tail -f /home/zhangshuyu/EnDeep4mC/log/app.log
    # 停止服务：pkill -f "app_5cv.py"