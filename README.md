# POPS: Postoperative pain prediction from preoperative EHR data

This is the official code repository for the manuscript "Development and prospective validation of postoperatve pain prediction from preoperative EHR data using ICD-10 and CPT attention-based set embeddings."

Code for algorithm development, evaluation, and statistical analysis is included in this repository.

Access to data used in this study requires a Data Use Agreement and IRB approval by the study institutions (MGB). Contingent upon these requirements, data are available from the authors upon reasonable request. Data directories have been excluded from this repository, but if available, can be softlinked as instructed.

## Setup and Dependencies

Clone the git repository:
```
git clone git@github.com:instigatorofawe/pain_prediction_pops.git
cd pain_prediction_pops
```

Create and activate a fresh conda environment:
```
conda create -n pops
conda activate pops
```

Install dependencies and install local project
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge rpy2
pip install pycox pytorch_lightning jupyter 
pip install -e .
```

## Softlinks to data files
```
ln -s [path to data directory] data
```
