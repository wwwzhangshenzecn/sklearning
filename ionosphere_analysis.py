import os
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split #训练集/测试集划分
from sklearn.neighbors import KNeighborsClassifier #KNN邻近算法
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.preprocessing import MinMaxScaler #标准预处理
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.externals import joblib #保存模型

def Compare():
    data_filename = os.path.join('Data','ionosphere.data')
    X = np.zeros((351, 34), dtype='float')
    y = np.zeros((351, ), dtype='bool')

    with open(data_filename, 'r') as file:
        reader = csv.reader(file)
        for i ,row in enumerate(reader):
            data = [float(dum) for dum in row[:-1]]
            X[i] = data
            y[i] = (row[-1] == 'g')


    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25 ,random_state=19)

    clf = RandomForestClassifier()
    clf =  joblib.load('RandomForestClassifiter.m')
    y_predictor_M = clf.predict(X_test)
    accuracy_M = np.mean(y_predictor_M == y_test) * 100
    print('RandomForestClassifiter-m :The accuracy is {0:.8f}%'.format(accuracy_M))


    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predictor = clf.predict(X_test)
    accuracy = np.mean(y_predictor == y_test) * 100
    print('DecisionTreeClassifiter :The accuracy is {0:.1f}%'.format(accuracy))



    clf_R = RandomForestClassifier()
    clf_R.fit(X_train, y_train )
    #joblib.dump(clf,'RandomForestClassifiter.m')
    y_predictor_R = clf_R.predict(X_test)
    accuracy_R = np.mean(y_predictor_R == y_test) * 100
    print('RandomForestClassifiter :The accuracy is {0:.1f}%'.format(accuracy_R))



    #导入K邻近分类器，并初始化,参数默认,选择邻近5个作为分类依据

    estimator = KNeighborsClassifier(n_neighbors=2) #估计器
    estimator.fit(X_train,y_train)
    y_predicted = estimator.predict(X_test)
    accuracy = np.mean(y_test == y_predicted) * 100
    print('KNN :The accuracy is {0:.1f}%'.format(accuracy))



Compare()



def data_pricessing():
#流水线在预处理中的作用
    data_filename= os.path.join('Data','ionosphere.data')
    X = np.zeros((351,34), dtype='float')
    y = np.zeros((351,), dtype='bool')

    with open(data_filename,'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = (row[-1] == 'g')
def MinMaxScale():

    data_filename= os.path.join('Data','ionosphere.data')
    X = np.zeros((351,34), dtype='float')
    y = np.zeros((351,), dtype='bool')

    with open(data_filename,'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = (row[-1] == 'g')


    X_broken = np.array(X)
    X_broken[:,::2] /= 10 #,前取行，后取列，：：2设置步长为2

    #标准预处理，将特征值的值域规范化为0-1，最小为0，最大为1
    estimator = KNeighborsClassifier()
    X_transformed = MinMaxScaler().fit_transform(X_broken)

    from sklearn.pipeline import Pipeline
    #创建流水线
    scaling_pipline = Pipeline([('scale',MinMaxScaler),('predict',KNeighborsClassifier())])
    scores = cross_val_score(scaling_pipline, X_broken , y, scoring='accuracy')
    print(np.mean(scores)*100)
#KNN算法与调参（邻近值）
def ionosphere():
    #导入数据
    #数据集每一行有35个数据，前34行为17座天线采集的数据，最后一额数据是g/b，好坏，是否提供了有价值的信息
    data_filename = os.path.join('Data','ionosphere.data')
    X = np.zeros((351, 34),dtype='float')
    y = np.zeros((351,), dtype='bool')


    with open(data_filename,'r') as input_file:
        reader = csv.reader(input_file)

        for i,row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = row[-1] == 'g'

    # print(X,y)
    #创建训练集和测试集
    X_transformed = MinMaxScaler().fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_transformed, y, random_state=14,test_size=0.5)

    #导入K邻近分类器，并初始化,参数默认,选择邻近5个作为分类依据

    estimator = KNeighborsClassifier(n_neighbors=2) #估计器

    estimator.fit(X_train,y_train)

    y_predicted = estimator.predict(X_test)
    accuracy = np.mean(y_test == y_predicted) * 100
    print('The accuracy is {0:.1f}%'.format(accuracy))

    #采用交叉验证，避免分割数据集造成的训练集的好差，减少运气成分

    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    average_accuracy = np.mean(scores) * 100
    print('The accuracy is {0:.1f}%'.format(average_accuracy))

    avg_scores = []
    all_scores =[]
    paramenter_values = list(range(1,51))

    for  n_neighbors in paramenter_values:
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring='accuracy')
        avg_scores.append(np.mean(scores))
        all_scores.append(scores)

    plt.plot(paramenter_values, avg_scores, '-o')
    plt.show() 

