import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from collections import defaultdict
classes = ['','Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def conver_number(x):
    try:
        return float(x)
    except:
        if(x == 'Iris-setosa'):
            return 1
        elif x == 'Iris-versicolor':
            return 2
        else:
            return 3

converteres = defaultdict(conver_number)
datasets = pd.read_csv('Data/iris/iris.data' ,header=None,converters=converteres )
print(datasets.ix[:5])

X = datasets[[0,1,2,3]].values
print(X[:5])
y = datasets[4].values
for i in range(len(y)):
    if y[i] == classes[1]:
        y[i] = int(1)
    elif y[i] == classes[2]:
        y[i] = int(2)
    elif y[i] == classes[3]:
        y[i] = int(3)
    else:
        y[i] = int(0)
y = np.array(y , dtype='int')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

parameter_space={
    'max_depth':[k for k in range(1,50)],
    'max_features':[1,2,3,4]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=25), parameter_space)
grid.fit(X_train,y_train)
x = [[4.1,4.2,4.8,3]]
y_predict = grid.predict(np.array(x))
print(y_predict)
print(grid.best_estimator_)