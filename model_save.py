from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# 保存模型
'''
注意：模型保存后会出现.npy文件，可能很多，预测的使用需要依赖这个文件，cp模型的时候需要一起cp，不然会报错。所以我选择使用pickle'''
import pickle

s = pickle.dumps(clf)
f = open('svm.model', 'w')
f.write(s)
f.close()

# 使用模型预测
f2 = open('svm.model', 'r')
s2 = f2.read()
clf2 = pickle.loads(s2)
clf2.predit(X, y)

# joblib