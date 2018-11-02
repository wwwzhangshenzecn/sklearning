import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA # 主成分分析算法
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

data_folder = os.path.join('Data', 'Advertisements', 'ad.data')

converters = defaultdict(int)
converters[1558] = lambda x:1 if x.strip() == 'ad.' else 0
ads = pd.read_csv(data_folder, header =None, converters=converters)
# 去掉pd里面的 ？ 值，或者除了ad. 的其他字符串，返回nan ,有效
ads = ads.applymap(lambda x: 0 if isinstance(x, str) and not x == "ad." else x)
X = ads.drop(1558, axis=1).values
y = ads[1558]

pca = PCA(n_components= 5)
Xd = pca.fit_transform(X)


X_train,X_test,y_train,y_test = train_test_split(Xd, y, test_size=0.5, random_state=14)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_predictor = clf.predict(X_test)
print('PCA: Accuracy:{:0.2f}%'.format(np.mean(y_predictor == y_test) * 100))


clf_pipeline = Pipeline([('predictor',DecisionTreeClassifier())])
clf_pipeline.fit(X_train,y_train)
print('Pipeline: Accuracy:{:0.2f}%'.format(np.mean(clf_pipeline.predict(X_test) == y_test) * 100))



