# Detecting personal attacks in the "wiki detox" dataset

This dataset is derived from the [Wiki Detox](https://meta.m.wikimedia.org/wiki/Research:Detox/Data_Release) datasets.

Text data with labels:
[text_data.zip](https://activelearning.blob.core.windows.net/activelearningdemo/text_data.zip)
Labels were aggregated from multiple crowdworkers; if half or more of the people who labelled a comment thought it was an attack, we mark it 1; otherwise it is 0.

Featurized dataset:
[attacke_use_encoded.Rds](https://altutorialweu.blob.core.windows.net/activelearningdemo/attacks_use_encoded.Rds)
