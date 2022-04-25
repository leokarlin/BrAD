# Title

# Installation
## Install Anaconda
## Create Environment
```shell
conda env create -f conda_env.yml
conda activate brad
```
## Full list of dependencies:
In case there are any problems installing the conda environment as describes above, the following is a full list of all 
dependecies need to run the training, testing and demo
1. pytorch (version ~ 1.8) and the corresponding torchvision
1. scikit-image
1. scikit-learn
1. tqdm
1. requests
1. jupyterlab (for demo)
1. ipywidgets (for demo)

# Data Prep
Please see [data_split/DATA_README.m](data_split/DATA_README.md)

# Downloading our pre-trained models
The model can be downloaded from: `https://drive.google.com/file/d/1T7v2xwAWQGsAv11-CEwKLmUH-TmCkue9/view?usp=sharing`

# Run Training
To run our model first activate the conda environment:
```shell
conda activate brad
```
run the main training script using torch.distributed.launch:

```shell
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad.py --data <DATA_ROOT>/clipart_train_test.txt,<DATA_ROOT>/painting_train_test.txt,<DATA_ROOT>/real_train_test.txt,<DATA_ROOT>/sketch_train_test.txt
```
Please see the config.py file for all available parameters or run:
```shell
python main_brad.py -h
```

# Run Test
To run our model first activate the conda environment:
```shell
conda activate brad
```
Run the main test script using torch.distributed.launch:

```shell
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad_test.py --resume <PATH_TO_TRAINED_MODEL> --src-domain <PATH_TO_SRC_DOMAIN_TXT_FILE> --dst-domain <PATH_TO_DST_DOMAIN_TXT_FILE> 
```
For instance, for 1-shot with source Real and target Painting use: 
```shell
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main_brad_test.py --resume <PATH_TO_TRAINED_MODEL> --src-domain <DATA_ROOT>/real_labeled_1.txt --dst-domain <DATA_ROOT>/painting_unlabeled_1.txt 
```

Use the flag --classifier to choose classifier type out of [retrieval, sgd, logistic], the default is retrieval.  
  
 

# Run Demo
## Initial setup
1. Make sure that the conda environment is set properly
1. Download the DomainNet Dataset
1. Download the pre-calculated features from https://drive.google.com/drive/folders/1OvowfDCNCxPCAgaOi0nVDEpiB3AF2Uut?usp=sharing
1. Run the jupyter notebook and open `demo.ipynb`
1. Modify the paths to the data under `data_root`
1. Run the demo section

