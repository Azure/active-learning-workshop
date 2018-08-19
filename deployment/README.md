# Deploying a model as a web service

Deployment is the process of exposing a predictive model to all eligible consumers.

These are scripts and example code related to deploying a scoring model as a web service.
Three methods of creating a web service are shown:
- Deploying a pure MLLib pipeline on MML-Spark
- 
- 

## Deployment of an MLLib model with MMLSpark

[MML Spark](https://github.com/Azure/mmlspark) is an open source data science platform library from Microsoft.

We demonstrate the creation and deployment of a text classification pipeline in the 
[TextClassificationWithMMLSpark](https://github.com/Azure/active-learning-workshop/blob/master/deployment/TextClassificationWithMMLSpark.ipynb)
notebook.

## Deployment of a scikit model with MMLSpark

We demonstrate the creation and deployment of a text classification pipeline in the 
[DeployTextServiceToMMLSpark](https://github.com/Azure/active-learning-workshop/blob/master/deployment/DeployTextServiceToMMLSpark.ipynb)
notebook.

The example preserves the scikit transform pipeline, which then has to be executed client-side.

## Deployment with Azure ML

Note: This method is not demonstrated at KDD 2018 because as of the time, Azure ML was not
an open-source software platform.

The training set was determined using active learning (in R); 
here we train a Python classifier on that training set, together with a preprocessing pipeline, 
and save it to disk. The serialized pipeline can be re-loaded and used to classify new cases.

The file `training_set_01.csv` contains the featurized cases that were selected in the active learning exercise. 
Here we just use that file to look up the ids of the selected training examples (in the column `rev_id`). Then we find the original text of these comments in `attack_data.csv`, which is one of the files in the zip file on [blob storage](https://activelearning.blob.core.windows.net/activelearningdemo/text_data.zip).
