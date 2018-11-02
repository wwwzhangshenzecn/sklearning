import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict

# 预测结果文件：src/step1/ground_truth/test_prediction.csv
num = []
def getPrediction():
    data_train = os.path.join('input', 'train.csv')
    data_test = os.path.join( 'input', 'test.csv')

    converters = defaultdict(int)
    ads = pd.read_csv(data_train, converters=converters)

    X = ads.drop('TARGET', axis=1).values
    y = ads["TARGET"].values
    ads_test = pd.read_csv(data_test)
    X_TEST = ads_test.values
    y_ID = ads_test['ID'].values

    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600, max_features = 3, subsample = 0.9)
    clf.fit(X, y)
    y_predictor = clf.predict_proba(X_TEST)

    sub_name = os.path.join('input', 'test_prediction.csv')
    with open(sub_name, 'w') as file:
        file.write('ID,TARGET\n')
        k = len(y_ID)
        for i in range(int(k)):
            line = str(y_ID[i]) + ',' + str(y_predictor[i][1])
            file.write(line + '\n')



getPrediction()