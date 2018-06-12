import pandas as pd #
import numpy as np
import re
import gensim
from sklearn.pipeline import Pipeline

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

