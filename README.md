# detailed paper work: [project report](https://drive.google.com/file/d/1WoNK3m2clUxOPQhgcbPMPqQlHoHKHI1A/view?usp=sharing)

#Introduction
In this project, we focus on a sub-task of fake news detection: [FNC]("http://www.fakenewschallenge.org/"). Given the headlines and body texts, the work is to classify the stance of body text w.r.t. the headlines into one of four categories: Agrees, Disagrees, Dicusses, and Unrelated.

We try to increase the prediction accuracy and F1 score by combining different related work. The pipeline contains feature extraction, dimensiona reduction of features, and a hybrid classification model. Using the CNN-LSTM architecture as the last part, we will combine different works of feature extraction and dimensional reduction to find out how the methods affect the classification results

The basic code architecture of W&B logger and Hydra part are from[MINSU3D](https://github.com/3dlg-hcvc/minsu3d)

# Setup
```shell
Environment requiremnets
* CUDA 11.X
* Python 3.8
Conda(recommended)
conda create -n stance-detection python=3.8
conda activate stance-detection

# install PyTorch 1.8.2
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia

# install Python libraries
pip install -e .

# install OpenBLAS and SparseHash via APT
sudo apt install libopenblas-dev libsparsehash-dev
python setup.py develop
```

### If there is still environment errors by pytorch, use the debugger to run the model or use the `minsu3d` conda environment and run `conda install nltk`

# Data Preparation
FNC-1 dataset
1. Downlolad the FNC-1 dataset from [FNC](https://github.com/FakeNewsChallenge/fnc-1), change the file name `competition_test_bodies.csv` to `test_bodies.csv`, change the file name `competition_test_stances.csv` to `test_stance.csv`

Pretrained model:
the pretrained model checkpoint for stance-detection-split1 is the `Stance-FNC-all_svd-best.ckpt` submitted in `output.zip`

# Training, Inference and Evalutaion
create the feature folders under the dataset dictionary
`mkdir {PATH_TO_FNC-1}/encoding extern_feat metadata stat_features`
# model combination before training and inference
Note: Configuration files are managed by Hydra, you can easily add or override any configuration attributes by passing them as arguments.\
to use different sub-model combination, modify the config file `config/model/stance.yaml`
```shell
use_external_feature: True
use_statistical_feature: True
use_cnn_lstm: True
use_svd: Tru
```
modify the first three lines to choose the combination of submodel, only four combinations is available:` "external_feature", "statistical_feature", "cnn_lstm", and "external_feature + statistical_feature + cnn_lstm"`, change `use_svd` to decide whether to use SVD method in CNN_LSTM model\

# config data set path
modify the line: config file `config/data/base.yaml`
`dataset_root_path: ${project_root_path}/data`
**as the root path to your download dataset, use relative path to project path**

# train a model from scratch
`python train.py model=stance-detetion data=fnc model.ckt_path={checkpoint_path} data.dataset_path={PATH_TO_FNC-1}`

# train a model from a checkpoint
`python train.py model=stance-detetion data=fnc model.ckt_path={pretrained_model_path} data.dataset_path={PATH_TO_FNC-1}`

# inference based on test set in split method 2 

modify the line: config file `config/model/stance.yaml`
`write_ouput: True`   to decide whether output the ground truth and prediction on test set during test step

run the following command to inference on test set
`python test.py model=stance-detetion data=fnc model.ckpt_path={pretrained_model_path} data.dataset_path={PATH_TO_FNC-1}`

# Reference
```
@ARTICLE{9178321,
  author={Umer, Muhammad and Imtiaz, Zainab and Ullah, Saleem and Mehmood, Arif and Choi, Gyu Sang and On, Byung-Won},
  journal={IEEE Access}, 
  title={Fake News Stance Detection Using Deep Learning Architecture (CNN-LSTM)}, 
  year={2020},
  volume={8},
  number={},
  pages={156695-156706},
  doi={10.1109/ACCESS.2020.3019735}}
  
```
