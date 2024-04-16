# 2DPhononINS-ML

## Getting Started

### Installation
1. Clone the repository
```
git clone https://github.com/suyaokun/2DPhononINS-ML.git
```
2. Create a virtual environment for the project
```
conda create -n 2DPhononINS python=3.9
```
```
conda activate 2DPhononINS
```
3. Install requirements
```
conda install --yes --file requirements.txt
```

### Data Downloads
The `data` and `dataset` are available at https://doi.org/10.5281/zenodo.10373288. Please download the data if you want to train your own model.

### Training
You can make changes to the code under `src` as needed to specify *folders*, *models*, *training paramenters*, etc. In each step, at least you need to change `autoencoder_dir`/`predictor_dir` to specify the folder you want to store or read your model.
#### Train the autoencoder with simulated data
```
python src/feature_extractor_training.py
```
#### Fine-tune the autoencoder with experimental data
```
python src/feature_extractor_finetuning.py
```
#### train the force constant regressor 
```
python src/regressor_training.py
```
### Inference
To see the model's predicted force constants on experimental (or simulated) spectra, run:
```
python src/inference.py
```

## Directory Structure

### data
This directory contains all the simulated spectra and labels.

### dataset
This directory contains the divided subsets for training, validation and testing purposes, as well as the experimental dataset.

### models
1. Models trained with different autoencoders, each in three training stages: simulated spectra only (e.g. FCVAE), fine-tuning with experimental spectra (e.g. FCVAE_FT), and extended training on simulated spectra (e.g. FCVAE_ext). 
2. Regressor network are stored under `models/xxx/predictor/1/`

### src
Source code.