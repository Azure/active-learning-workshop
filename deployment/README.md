These are scripts and example code related to deploying a scoring model as a web service.

The training set was determined using active learning (in R); here we train a Python classifier on that training set, together with a preprocessing pipeline, and save it to disk. The serialized pipeline can be re-loaded and used to classify new cases.
