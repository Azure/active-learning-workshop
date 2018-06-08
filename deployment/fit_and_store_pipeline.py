# cd /c/Users/rhorton/Documents/conferences/MLADS/MLADS_spring_2018
 
import pandas as pd
import numpy as np #
import re #
import random
import gensim # 
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from pipeline_parts import *

###

training_set_file = "training_set_01.csv"
text_data_file = "attack_data.csv"
w2v_file = 'glove_6B_50d_w2v.txt' # convert glove file to w2v format using gensim.scripts.glove2word2vec
# glove_file = 'glove.6B.50d.txt'
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_file, w2v_file)

word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

text_data = pd.read_csv(text_data_file, encoding='windows-1252')
text_data = text_data.set_index("rev_id")

training_set_rev_ids = pd.read_csv(training_set_file).rev_id
test_candidate_rev_ids = set.difference(set(text_data.index.values), set(training_set_rev_ids))
test_set_rev_ids = random.sample(test_candidate_rev_ids, 1000)

training_data = text_data.loc[training_set_rev_ids]
test_data = text_data.loc[test_set_rev_ids]


model_pipeline = Pipeline([
    ('preprocessor', GensimPreprocessor()),
    ('vectorizer', AvgWordVectorFeaturizer(embedding=word_vectors)),
    ('classifier', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)),
])

fitted_model = model_pipeline.fit(training_data.comment, [int(x) for x in training_data.is_attack])

pred = fitted_model.predict(test_data.comment)

confusion_matrix([int(x) for x in test_data.is_attack], pred)

joblib.dump(fitted_model, 'rf_attack_classifier_pipeline.pkl') 


###
