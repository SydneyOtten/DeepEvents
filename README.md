# DeepEvents

This repository contains the code and trained networks to reproduce the main results of https://arxiv.org/abs/1901.00875: a deep generative model that creates events from a complex physical process: ttbar production with MET, METphi and the 4-vectors of up two six final state objects: four jets and up to two leptons.

The script called 'run.py' that is contained in the main directory is the main piece of this repository. The files contained in 'traditional_methods', 'models' and 'results' are auxiliary material in case someone wishes to reproduce or see more plots of all studies in the article above. Here we focus on the model we find to deliver the best performance: the B-VAE trained with a 20-dimensional latent space, B=10**(-6), alpha=1, gamma=0.05. In this README, we will show you how to a) train this model and b) provide you with a pretrained model and show you how to use it. 

# System Requirements

The script has been tested on Windows 7 and Ubuntu 16.04.

Running the scripts was tested using:
numpy 1.17.4
scikit-learn 0.22
pandas 0.25.3
tensorflow-gpu 1.14.0
Keras 2.2.5

# Installation Guide

Once all required packages mentioned above are installed, no further installation steps need to be taken.

# Instructions to run

a) Training the model

1. Download the ttbar dataset from zenodo: https://zenodo.org/record/3560661#.XebwfOhKiUk
2. Place the dataset 'ttb.csv' in the main directory.
3. Type 'python run.py' into your command line interface.

The output will be .hdf5 and .h5 files containing the trained model and a file containing 1.2 million events using the density information buffer and gamma=0.05.

We recommend you to train the model on a GPU. The training time on a GTX970 is approximately 30 minutes.

In case you wish to train the B-VAE with other settings, you have to change the following in 'run.py':
to change
- the number of samples to train on, adjust line 58 'trainsize = ...'. (default is 100000)
- latent dimension, adjust line 75
- B, adjust line 152
- alpha and gamma, adjust line 213

b) Using the pretrained model

1. Download the ttbar dataset from zenodo: https://zenodo.org/record/3560661#.XebwfOhKiUk
2. Place the dataset 'ttb.csv' in the main directory.
3. Type 'python run.py -w ttbar_20d_e-6.hdf5' into your command line interface.

The expected output is a file containing 1.2 million ttbar events created by the B-VAE within a few minutes.

# Evaluation

If you are interested in the values for the two figures of merit defined in the abovementioned article, place the generated events in the folder 'Evaluation' and type 'python FoM.py' into your CLI. In case the name of the file containing the events you want to evaluate is not 'B-VAE_events.csv', you need to adjust line 152 in FoM.py.
