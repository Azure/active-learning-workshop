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
    joblib.dump(fitted_model, 'rf_attack_classifier_pipeline.pkl') 
    return fitted_model

def evaluate(fitted_model, test_data):
    pred = fitted_model.predict(test_data.comment)
    return confusion_matrix([int(x) for x in test_data.is_attack], pred)

##############################
## scoring script business

global reloaded_model

def init():
    """
    Init function of the scoring script
    """
    reloaded_model = joblib.load('rf_attack_classifier_pipeline.pkl')

def run(phrase_list):
    return reloaded_model.predict(phrase_list)

##############################

def script_main():

    training_data, test_data = create_train_test_split()
    fitted_model = train(training_data)
    print(evaluate(fitted_model, test_data))
    
    ### score script
    init()
    p = run(['You are scum.', 'I like your shoes.', 'You are pxzx.', 
             'Your mother was a hamster and your father smelt of elderberries',
             'One bag of hagfish slime, please'])
    print(p)


