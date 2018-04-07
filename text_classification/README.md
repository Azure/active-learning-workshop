# Detecting personal attacks in the "wiki detox" dataset

This dataset is derived from the [Wiki Detox](https://meta.m.wikimedia.org/wiki/Research:Detox/Data_Release) datasets.

Text data with labels:
[text_data.zip](https://activelearning.blob.core.windows.net/activelearningdemo/text_data.zip)
Labels were aggregated from multiple crowdworkers; if half or more of the people who labelled a comment thought it was an attack, we mark it 1; otherwise it is 0.

Featurized dataset:
[featurized_wiki_comments_attack_feather.zip](https://activelearning.blob.core.windows.net/activelearningdemo/featurized_wiki_comments_attack_feather.zip)

Featurized dataset in zipped CSV format (in case you want to look at it in Excel):
[featurized_wiki_comments_attack.zip](https://activelearning.blob.core.windows.net/activelearningdemo/featurized_wiki_comments_attack.zip)

Glove vectors for the subset of glove.6B words that are used in this dataset:
[wiki_wordvecs_50d.csv](https://activelearning.blob.core.windows.net/activelearningdemo/wiki_wordvecs_50d.csv)
Each row is a word, and there are 50 columns representing the features. These rows are filtered from the full word embedding file (glove.6B.50d.txt, available from http://nlp.stanford.edu/data/glove.6B.zip), to only include those words that actually appear in our corpus. These are the "6B" embeddings, trained on 6 billion tokens from Wikipedia + Gigaword 5.

