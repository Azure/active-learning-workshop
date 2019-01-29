# Detecting personal attacks in the "wiki detox" dataset

This dataset is derived from the [Wiki Detox](https://meta.m.wikimedia.org/wiki/Research:Detox/Data_Release) datasets.

Text data with labels:
[text_data.zip](https://activelearnwestus.blob.core.windows.net/activelearningdemo/text_data.zip)
Labels were aggregated from multiple crowdworkers; if half or more of the people who labelled a comment thought it was an attack, we mark it 1; otherwise it is 0.

Featurized dataset:
[featurized_wiki_comments_attack data](https://bobpubdata.blob.core.windows.net/pub/wiki_attacks_use_encoded_30k.feather.zip)
This is a zipped file in [feather](https://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/) format.
