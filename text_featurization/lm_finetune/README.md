# Contextualized Word Vectors and Language Models

## Setup

These notebooks are designed to use the provided [conda environment](./conda.yml). If you're using a provided DSVM, this environment should have already been created for you as part of the `setup.sh` script. If you're running from home, you can create the conda environment by running: `conda env create -f conda.yml`. 

## Bag of Words vs Language Model Encoder Vectors

The notebook [1A-Toxic_BoW_vs_LM.ipynb](./1A-Toxic_BoW_vs_LM.ipynb) provides a comparison of the discriminative power of pre-trained GloVe vectors vs pre-trained language model encoder vectors for predicting abusive comments in online forums.

## Zero Shot Learning

The notebook [1B-Transformer-LM-ZeroShot.ipynb](./1B-Transformer-LM-ZeroShot.ipynb) describes how to use language model for zero-shot learning in four different datasets.