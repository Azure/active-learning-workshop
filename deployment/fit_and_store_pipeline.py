# cd /c/Users/rhorton/Documents/conferences/MLADS/MLADS_spring_2018
 
import pandas as pd
import numpy as np
import re
import random
import gensim
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


class GensimPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, newline_token='NEWLINE_TOKEN'):
        self.newline_pat = re.compile(newline_token)
    
    def fit(self, X, y=None):
        return self
    
    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]
    
    def transform(self, X):
        return [ list(self.tokenize(txt)) for txt in X ]
    
    def tokenize(self, doc):
        doc = self.newline_pat.sub(' ', doc)
        return gensim.utils.simple_preprocess(doc)


class AvgWordVectorFeaturizer(object):
    def __init__(self, embedding):
        self.embedding = embedding
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        # X is a list of tokenized documents
        return np.array([
            np.mean([self.embedding[t] for t in token_vec if t in self.embedding]
                    or [np.zeros(self.embedding.vector_size)], axis=0)
            for token_vec in X
        ])

###

def create_small_w2v_file():
    glove_file = 'E:/Projects/MLADS18S/glove.6B.50d.txt'
    # convert glove file to w2v format using gensim.scripts.glove2word2vec
    w2v_file = './glove_6B_50d_w2v.txt' 
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, w2v_file)

def create_train_test_split():
    
    training_set_file = "training_set_01.csv"
    text_data_file = "attack_data.csv"

    word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    text_data = pd.read_csv(text_data_file, encoding='windows-1252')
    text_data = text_data.set_index("rev_id")

    training_set_rev_ids = pd.read_csv(training_set_file).rev_id
    test_candidate_rev_ids = set.difference(set(text_data.index.values), set(training_set_rev_ids))
    
    # take a random sample of 1000 ids for test set
    test_set_rev_ids = random.sample(test_candidate_rev_ids, 1000)
    
    training_data = text_data.loc[training_set_rev_ids]
    test_data = text_data.loc[test_set_rev_ids]
    return training_data, test_data

def train():
    """
    Creates a pickled trained model.
    """

    model_pipeline = Pipeline([
        ('preprocessor', GensimPreprocessor()),
        ('vectorizer', AvgWordVectorFeaturizer(embedding=word_vectors)),
        ('classifier', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)),
    ])

    fitted_model = model_pipeline.fit(training_data.comment, [int(x) for x in training_data.is_attack])
    return fitted_model

    pred = fitted_model.predict(test_data.comment)
    confusion_matrix([int(x) for x in test_data.is_attack], pred)
    
    joblib.dump(fitted_model, 'rf_attack_classifier_pipeline.pkl') 


def init():
    """
    Init function of the scoring script
    """"

    reloaded_model = joblib.load('rf_attack_classifier_pipeline.pkl')


def run():
    reloaded_model.predict(['You are scum.', 'I like your shoes.', 'You are pxzx.'])
