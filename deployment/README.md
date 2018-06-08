# Deploying a model as a web service

These are scripts and example code related to deploying a scoring model as a web service.

The training set was determined using active learning (in R); here we train a Python classifier on that training set, together with a preprocessing pipeline, and save it to disk. The serialized pipeline can be re-loaded and used to classify new cases.

The file `training_set_o1.csv` contains the featurized cases that were selected in the active learning exercise. Here we just use that file to look up the ids of the selected training examples (in the column `rev_id`). Then we find the original text of these comments in `attack_data.csv`, which is one of the files in the zip file on [blob storage](https://activelearning.blob.core.windows.net/activelearningdemo/text_data.zip).
