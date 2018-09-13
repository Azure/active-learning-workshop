# cd /c/Users/rhorton/Documents/conferences/MLADS/MLADS_spring_2018

import pandas as pd #
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

training_set_file = "training_set_01.csv"
test_set_file = "test_set_01.csv"

training_data = pd.read_csv(training_set_file)
test_data = pd.read_csv(test_set_file)
input_data = training_data.iloc[:,1:51]

model = RandomForestClassifier(criterion="entropy", n_estimators=1000)
fitted_model = model.fit(input_data, [int(x) for x in training_data.flagged])



pred_class = fitted_model.predict(test_data.iloc[:,1:51])

from sklearn.metrics import roc_auc_score
pred_prob = fitted_model.predict_proba(test_data.iloc[:,1:51])
scores = pred_prob[:,1]
auc = roc_auc_score([int(x) for x in test_data.flagged], scores)
print(auc)

joblib.dump(model, 'attack_model.pkl') 


###
