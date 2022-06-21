# XENIA


### The repository contains code refered to the work:

Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba



![XENIA](!![image](https://user-images.githubusercontent.com/38468857/174758214-12e27a42-5347-450c-bc14-31ae0e95758c.png)





### Code Requirements

 * [Python 3.9](https://www.python.org/downloads/release/python-390/)
 * [Keras 2.7](https://github.com/keras-team/keras)
 * [Tensorflow 2.7](https://www.tensorflow.org/)
 * [Scikit learn](https://scikit-learn.org/stable/)
 * [Matplotlib 3.5](https://matplotlib.org/)
 * [Pandas 1.3.5](https://pandas.pydata.org/)
 * [Numpy 1.19.3](https://numpy.org/)
 * [Dalex 1.4.1](https://github.com/ModelOriented/DALEX)
 * [adversarial-robustness-toolbox 1.9](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
 * [scikit-learn-extra 0.2.0](https://scikit-learn-extra.readthedocs.io/en/stable/)
 * [Hyperopt 0.2.5](https://pypi.org/project/hyperopt/)


###  Description for this repository

* Datasets and Models.
   [Datasets and Models](https://drive.google.com/drive/folders/1TUCquwp-mB4UVcJrnoHkRQKkCKKJ407n).
  
   

### How to use

The implementation for all the experiments used in this work are listed in this repository.
  * main.py : to run XENIA
 


## Replicate the Experiments

To replicate the experiments of this work, the models and datasets that have been saved in [Datasets and Models](https://drive.google.com/drive/folders/1KnGoU2l1dxVQxIpq8AX9dAzTHuCl-_I5) can be used. Global Variable are saved in PANACEA.conf :

* ###### TRAIN_BASELINE = 0   &emsp;        #1 train baseline with hyperopt <br />
* ###### CREATE_ADVERSARIAL_SET=0 &emsp;  #if 1 create the adversarial samples <br />
* ###### NUMBER_OF_MODELS=5       &emsp;  #Number of Models to be chosen <br />

* ###### sigma=0.05             &emsp;      #percentage of adversarial samples <br />
* ###### CREATE_CANDIDATE_MODELS = 0  &emsp; #if 1 create the candidate models with the number of NUMBER_OF_MODELS <br />
* ###### USE_MEDOIDS=1              &emsp;  #if 0 the execution is performed with ensemble without medoids <br />
* ###### CREATE_DALEX_DATASET= 0  &emsp;    #if 1 create the csv with dalex values if 0 the csv file will be loaded <br />
 
* ###### ENSEMBLE = 1       &emsp;          #To choose the models based on the XAI and Clustering <br />
* ###### TRAIN_ENSEMBLE = 0   &emsp;        #1 To choose the ensemble members based on XAI and Clustering, and to train the ensemble model using hyperopt, #0 to compute the majority voting for the ensemble model <br />

* ###### INDIVIDUAL_PREDICTION=0   &emsp;   #0 no prediction, 1 prediction for individual models <br />

* ###### SEQUENTIAL = 0          &emsp;     #to choose the models sequentially <br />
* ###### USE_SEQUENTIAL = 0     &emsp;      #1 to train the ensemble model using hyperopt, 0 to compute the majority voting <br />

* ###### Majority_Voting = 0    &emsp;      #1 To choose Majority Voting for Predicting the Models <br />
* ###### Medoid_Voting = 0     &emsp;       #1 for choosing majority voting for medoids, 0 for choosing majority voting for sequential models <br />


