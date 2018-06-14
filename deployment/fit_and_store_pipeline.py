# cd /c/Users/rhorton/Documents/conferences/MLADS/MLADS_spring_2018

import numpy as np
import pandas as pd
import re
import random
import gensim 
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from sklearn.base import BaseEstimator, TransformerMixin

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
    def __init__(self, embedding, alpha=0, restrict_vocab=400000):
        self.embedding = embedding
        self.word2index = { w:i for i,w in enumerate(embedding.index2word) }
        self.restrict_vocab = restrict_vocab
        self.alpha = alpha
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        # X is a list of tokenized documents
        return np.array([
            np.mean([self.embedding[t] for t in token_vec 
                        if t in self.embedding and 
                        (self.word2index[t] < self.restrict_vocab) and
                        (np.max(np.absolute(self.embedding[t])) > self.alpha)
                    ]
                    or [np.zeros(self.embedding.vector_size)], axis=0)
            for token_vec in X
        ])

###

data_files_dir = 'E:/Projects/MLADS18S/'

def create_small_w2v_file():
    # Location of source file on Tomas' computer
    glove_file = data_files_dir + 'glove.6B.50d.txt'

    # create this file in current wd
    w2v_file = './glove_6B_50d_w2v.txt' 
    
    # convert glove file to w2v format using gensim.scripts.glove2word2vec
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, w2v_file)

def unzip_file_here(f):
    import zipfile
    zip_ref = zipfile.ZipFile(f, 'r')
    zip_ref.extractall('.')
    zip_ref.close()

def create_train_test_split():
    
    # Location of original file of "attacks" on Tomas' computer
    text_data_file = 'attack_data.csv'
    # this is the pre-featurized subset of the attacks
    training_set_file = 'training_set_01.csv'

    text_data = pd.read_csv(text_data_file, encoding='windows-1252')
    text_data = text_data.set_index("rev_id")

    training_set_rev_ids = pd.read_csv(training_set_file).rev_id
    test_candidate_rev_ids = set.difference(set(text_data.index.values), set(training_set_rev_ids))
    
    # take a random sample of 1000 ids for test set
    test_set_rev_ids = random.sample(test_candidate_rev_ids, 1000)
    
    training_data = text_data.loc[training_set_rev_ids]
    test_data = text_data.loc[test_set_rev_ids]
    return training_data, test_data

def train(training_data, w2v_file = './miniglove_6B_50d_w2v.txt'):
    """
    Creates a pickled trained model.
    """

    word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    model_pipeline = Pipeline([
        ('preprocessor', GensimPreprocessor()),
        ('vectorizer', AvgWordVectorFeaturizer(embedding=word_vectors)),
        ('classifier', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)),
    ])

    fitted_model = model_pipeline.fit(training_data.comment, [int(x) for x in training_data.is_attack])
    joblib.dump(fitted_model, 'rf_attack_pipeline.pkl') 
    return fitted_model

def evaluate(fitted_model, test_data):
    pred = fitted_model.predict(test_data.comment)
    return confusion_matrix([int(x) for x in test_data.is_attack], pred)

##############################
## scoring script business

def init():

    from sklearn.externals import joblib

    global reloaded_model
    """
    Init function of the scoring script
    """
    reloaded_model = joblib.load('rf_attack_pipeline.pkl')

def run(raw_data):

    import json

    try:
        phrase_list = json.loads(raw_data)['data']
        result = reloaded_model.predict(phrase_list)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})

##############################

test_attacks = ['You are scum.', 'I like your shoes.', 'You are pxzx.', 
             'Your mother was a hamster and your father smelt of elderberries',
             'One bag of hagfish slime, please']

def script_main():

    import json

    training_data, test_data = create_train_test_split()
    fitted_model = train(training_data)
    print(evaluate(fitted_model, test_data))
    
    ### score script
    init()
    encoded_data = bytes(json.dumps({"data": test_attacks}), encoding = 'utf8')

    p = run(encoded_data)
    print(p)


