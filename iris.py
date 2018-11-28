from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
iris = load_iris()
X = iris.data
y = iris.target
# print(X, y)

from sklearn import svm

from sklearn.ensemble import BaggingClassifier,RandomForestClassifier

n = 100
acb = []
acr = []
for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    bcf = BaggingClassifier(n_estimators=10+2*i,max_features=4,n_jobs=-1)
    rcf = RandomForestClassifier(n_estimators=10+2*i,max_features=4,n_jobs=-1)

    bcf.fit(X_train,y_train)
    rcf.fit(X_train,y_train)

    acb.append(sum(bcf.predict(X_test) != y_test)/len(y_test))
    acr.append(sum(rcf.predict(X_test) != y_test)/len(y_test))

plt.plot(range(1,n+1), acb, 'r-',label='Bagging')
plt.plot(range(1,n+1), acr, 'g-',label='RandomForest')

plt.legend()
plt.show()
