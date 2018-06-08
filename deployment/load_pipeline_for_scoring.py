
from sklearn.externals import joblib
from pipeline_parts import *

reloaded_model = joblib.load('rf_attack_classifier_pipeline.pkl')
reloaded_model.predict(['You are scum.', 'I like your shoes.', 'You are pxzx.'])
