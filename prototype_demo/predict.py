import sys
import argparse
import json
import pprint
import pandas as pd
import numpy as np
import utils as ut 
import joblib
import os
from collections import Counter

class CatchOutErr:
    def __init__(self):
        self.value = ''
    def write(self, txt):
        self.value += txt

catchOutErr = CatchOutErr()
sys.stdout = catchOutErr
sys.stderr = catchOutErr

# RESULTS_DIR = "data/results"
# RF = "data/results/classifiers/RandomForestClassifier.pkl"
# SVC = "data/results/classifiers/LinearSVC.pkl"
# TEST = "../data/test/unlabeled/unlabeled_merged_data.json"

def main(modelFullPath, instrLineInfos):   
    # saved_data = os.path.join(RESULTS_DIR, 'test_data.pkl')
    print(instrLineInfos)
    df = pd.DataFrame()
    # try:
    #     df = pd.read_pickle(saved_data)
    # except FileNotFoundError as e: 
    #     print (e)

    try:
        clf = joblib.load(modelFullPath) # path to .pkl file
        print("Estimator found in {}:\n {}".format(modelFullPath, clf))
    except FileNotFoundError as e: 
        print (e)

    if df.empty:
        json_data = json.loads(instrLineInfos)
        unlabeled_data = ut.create_df_rows_from_instructions(json_data)
        df = pd.DataFrame(unlabeled_data).astype(int)
        # df.to_pickle(saved_data)

    print("Result probabilities for model: {}".format(clf))
    print(df)
    df = df.iloc[:, :-1]

    if hasattr(clf, 'predict_proba'):
        pp = clf.predict_proba(df)
        predict_prob = pd.DataFrame(pp, columns=clf.classes_)
        predict = predict_prob.idxmax(axis=1)
        c = Counter(predict)
        print("Vulnerabilities found  {} / {}".format(sum(c.values()) - c[0], sum(c.values())))
        print(c)
    else:
        predict = clf.predict(df)
        print("Vulnerabilities found  {} / {}".format(np.count_nonzero(predict), len(predict)))

    new_df = pd.DataFrame(data=predict, columns=['label'])
    new_df['instr_no'] = new_df.index

    result_json = new_df.to_json(orient='records')

    return result_json

# if __name__ == "__main__":
#     main(SVC, None)