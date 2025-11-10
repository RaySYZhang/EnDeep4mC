import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

# Path Config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import read_fasta_data, ml_code
from feature_engineering.feature_selection import get_feature_methods

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file upload to 16MB

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU has been detected, and the on-demand growth of video memory has been enabled")
    except RuntimeError as e:
        print(e)

# Global Configs
SPECIES_LIST = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
                '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
MODELS_DIR = os.path.join(project_dir, 'pretrained_models', 'indiv')

# Initialize thread/process pool
model_executor = ThreadPoolExecutor(max_workers=4)
feature_executor = ProcessPoolExecutor(max_workers=4)

# Model cache
model_cache = {}

class ModelLoader:
    """Enhanced model loader, supporting dynamic feature method reconstruction"""
    @staticmethod
    def load_ensemble(species):
        return joblib.load(os.path.join(MODELS_DIR, f'ensemble_indiv_{species}.pkl'))
    
    @staticmethod
    def load_dl_model(model_name, species):
        return tf.keras.models.load_model(
            os.path.join(MODELS_DIR, f'{model_name.lower()}_{species}.h5')
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
        return joblib.load(os.path.join(MODELS_DIR, f'{model_name}_{species}_scaler.pkl'))

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
    """Enhanced sequence processor, retaining the original title"""
    @staticmethod
    def _parse_fasta(content):
        sequences = []
        current_seq = []
        seq_id = None  # Initialize as None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                # Save the current sequence when encountering a new title
                if current_seq:
                    sequences.append((''.join(current_seq), seq_id))
                    current_seq = []
                # Extract the complete title line (retain all content after>, up to 100 characters)
                seq_id = line[1:].strip()[:100]
            elif line:
                current_seq.append(line.upper())
        
        # Process the last sequence
        if current_seq:
            sequences.append((''.join(current_seq), seq_id))
        
        # Keep the first appearing title when removing duplicates
        seen = {}
        for seq, seq_id in sequences:
            if seq not in seen:
                seen[seq] = seq_id
        return [(seq, seen[seq]) for seq in seen]

    def process_input(self, text_input, file_input):
        all_seqs = []
        
        if text_input.strip():
            all_seqs += self._parse_fasta(text_input)
        if file_input and file_input.filename:
            try:
                content = file_input.read().decode('utf-8')
                all_seqs += self._parse_fasta(content)
            except UnicodeDecodeError:
                raise ValueError("File encoding error: Only UTF-8 encoding is supported")
        
        # Keep the original title when building a DataFrame
        return pd.DataFrame(
            [{
                'seq_id': seq_id or "Untitled_Sequence",  # Ensure that the title is not empty
                'seq': seq,
                'label': 0,
                'is_train': False
            } for seq, seq_id in all_seqs]
        ) if all_seqs else pd.DataFrame()

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
            if row['seq'] in self.cache:
                cached_features.append(self.cache[row['seq']])
            else:
                new_sequences.append(row.to_dict())
        
        # Parallel processing of new data
        if new_sequences:
            gen_func = partial(self._generate_single, 
                             feature_methods=self.feature_methods,
                             feature_indices=self.feature_indices)
            results = list(feature_executor.map(gen_func, new_sequences))
            
            # Dimension verification
            expected_dim = len(self.feature_indices)
            for data, feat in zip(new_sequences, results):
                if feat.shape[0] != expected_dim:
                    raise ValueError(f"Abnormal feature dimension: {data['seq'][:50]}...")
                self.cache[data['seq']] = feat
                cached_features.append(feat)
        
        features = np.vstack(cached_features)
        
        # Final standardization
        if features.shape[1] != self.scaler.n_features_in_:
            raise ValueError(f"Feature dimension mismatch: {self.scaler.n_features_in_} vs {features.shape[1]}")
        
        return self.scaler.transform(features)
    
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
            return jsonify({'error': 'No effective DNA sequences were found'}), 400
        
        models = load_species_models(species)
        
        meta_features = []
        for model_name in BASE_MODELS:
            generator = FeatureGenerator(
                feature_config=models['feature_configs'][model_name],
                scaler=models['scalers'][model_name]
            )
            X = generator.generate(input_df)
            
            dl_model = models['base_models'][model_name]
            dl_input = X.reshape(-1, 1, X.shape[1])
            pred_proba = dl_model.predict(dl_input, verbose=0).flatten()
            meta_features.append(pred_proba)
        
        meta_X = np.column_stack(meta_features)
        probabilities = models['models'].predict_proba(meta_X)[:, 1]
        
        results = [{
            'seq_id': row['seq_id'],  # Return to the original title
            'sequence': str(row['seq']),
            'probability': float(round(prob, 4)),
            'is_4mC_site': bool(prob >= 0.5)
        } for prob, (_, row) in zip(probabilities, input_df.iterrows())]
        
        return jsonify({'results': results})
    
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
    app.run(host='0.0.0.0', port=5000, threaded=True)