# EnDeep4mC: A dual-adaptive feature encoding framework in deep ensembles for predicting DNA N4-methylcytosine sites

## Overview
EnDeep4mC is a deep learning framework for predicting DNA N4-methylcytosine sites through dual-adaptive feature encoding and ensemble learning. This repository contains the complete implementation, pre-trained models, and evaluation scripts.

## Quick Links
- **Web Server**: http://112.124.26.17:7012
- **Source Code**: https://github.com/ShuyuZhang0115/EnDeep4mC

## System Requirements

### Hardware Specifications
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended  
- **Storage**: 2GB for models and dependencies

### Software Dependencies
- Python 3.7+
- TensorFlow 2.4+
- Flask 2.0+
- scikit-learn 1.0+
- Joblib 1.1+

## Installation & Configuration
```bash
conda env create -f environment.yaml
```

### Runtime Configuration
```python
# Server settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit
model_executor = ThreadPoolExecutor(max_workers=4)    # Model inference threads
feature_executor = ProcessPoolExecutor(max_workers=4) # Feature generation processes

# Supported species and models
SPECIES_LIST = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
                '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
```

## data/4mC
The dataset we used in this experiment is derived from the 4mC dataset constructed in the work of EpiTEAmDNA. For details, refer to：
```bash
Li F, Liu S, Li K, Zhang Y, Duan M, Yao Z, Zhu G, Guo Y, Wang Y, Huang L, Zhou F. EpiTEAmDNA: Sequence feature representation via transfer learning and ensemble learning for identifying multiple DNA epigenetic modification types across species. Comput Biol Med. 2023 Jun;160:107030. doi: 10.1016/j.compbiomed.2023.107030. Epub 2023 May 11. PMID: 37196456.
```

## evaluations
Evaluation results of various experiments.
- ablation_feature: Design and results of feature ablation experiments.
- ablation_model_indiv: Design and results of model ablation experiments (using independent test sets).
- cross_predict: Results of cross-species prediction experiment.
- kmer_analysis: Design and Results of kmer spectrum analysis experiment for eukaryotes/prokaryotes.

## feature_engineering
The feature engineering module, which can be transferred to the feature selection & encoding process of other deep learning models.
- fea_index.py: Performance quantification functions for 14 features across 6 species.
- fea_index_extra.py: Performance quantification functions for 14 features across 10 supplementary species.
- feature_selection.py: Feature selection support functions (6 species).
- feature_selection_extra.py: Feature selection support functions (10 supplementary species).
- ifs_on_base_models.py: Species-model joint incremental feature selection function (6 species).
- ifs_on_base_models_extra.py: Species-model joint incremental feature selection function (10 supplementary species).
- fea_index: Stores the results of running fea_index.py.
- fea_index_extra_species: Stores the results of running fea_index_extra.py.
- ifs_result: Stores the results of running ifs_on_base_models.py.
- ifs_result_extra_species: Stores the results of running ifs_on_base_models_extra.py.

## fs
Contains detailed definitions of several feature encoding methods from the biological tool iLearn. We mainly referenced the definitions of 14 candidate feature encoding methods from the open-source code of EnDeep4mC mentioned above.

## log
Log files of main experiments.

## models
Definitions of the deep learning models proposed in this study.
- deep_models: Contains definitions of 3 deep learning base classifiers.
- pretrain_ensemble_model_5cv: Definitions and pre-training scripts for stacked ensemble models (5-fold cross-validation).
- pretrain_ensemble_model_indiv: Definitions and pre-training scripts for stacked ensemble models (independent test set validation).

## prepare
Definitions for pre-training base models. The files prepare_dl.py and prepare_ml.py are from the open-source code of the EnDeep4mC work. We referenced the data processing functions for DNA sequences.
- prepare_dl.py: A configuration file related to deep learning in the EnDeep4mC project. This experiment mainly uses some of its data processing functions.
- prepare_ml.py: A configuration file related to machine learning in the EnDeep4mC project. This experiment mainly uses some of its data processing functions.
- pretrain_base_models_5cv: Pre-training scripts for base models with dynamic feature selection strategies across 6 species (5-fold cross-validation).
- pretrain_base_models_indiv: Pre-training scripts for base models with dynamic feature selection strategies across 6 species (independent test set validation).

## pretrained_models
Stores pre-trained models (.h5) from the prepare and models modules.
- 5cv: Stores all pre-trained models using 5-fold cross-validation.
- indiv: Stores all pre-trained models using independent test set validation.

## tools
Some supplementary tools involved in this study, only used for evaluation testing and auxiliary analysis, not actually used in our work.
- CD-HIT.py: Script for removing DNA sequence redundancy using the CD-HIT tool.
- GC_content.py: Script for evaluating the GC content in DNA sequences.
- diversity_metrics.py: Script for evaluating the diversity metrics of base models.
- motif.py: Script for finding motifs in DNA sequences using the meme tool.
- tools.py: Originally supplementary tools in the EnDeep4mC project, including functions such as t-sne visualization.

## web_server
A web server built based on the proposed EnDeep4mC model, which can be used online by users.

## Software Availability
- Web Server Access: http://112.124.26.17:7012
- Source Code Repository: https://github.com/ShuyuZhang0115/EnDeep4mC
- The live web server provides immediate access to the prediction functionality, while the GitHub repository contains complete implementation for local deployment and methodological transparency. Both resources include comprehensive documentation supporting research reproducibility and framework extension.
