import os
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest # 使用SelectKBase转换器，然后用卡方函数打分
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr    # 皮尔逊相关系数
warnings.filterwarnings('ignore')

data_folder = os.path.join('Data','Adult')
adult_filename = os.path.join(data_folder, 'adult.data')
adult_filename = os.path.join(data_folder, "adult.data")
adult = pd.read_csv(adult_filename, header=None, names=["Age", "Work-Class", "fnlwgt", "Education",
                                                        "Education-Num", "Marital-Status", "Occupation",
                                                        "Relationship", "Race", "Sex", "Capital-gain",
                                                        "Capital-loss", "Hours-per-week", "Native-Country",
                                                        "Earnings-Raw"])
adult.dropna(how='all', inplace=True) # 数据最后两行是空行，也会作为数据存在，删除无效数据
adult['LongHours'] = adult['Hours-per-week'] > 40
X = adult[['Age', 'Education-Num', 'Capital-gain', 'Capital-loss', 'Hours-per-week']].values
y = (adult['Earnings-Raw'] == ' >50K').values

# K个最佳返回特征选择器
transformer = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer.fit_transform(X, y)


# pearsonr只能处理一维数组，包装器函数处理多维数组
def multivariate_pearsonr(X,y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

transformer_p = SelectKBest(score_func=multivariate_pearsonr, k=2)
Xt_pearson = transformer_p.fit_transform(X,y)

clf = DecisionTreeClassifier(random_state=14)
clf.fit(Xt_chi2,y)
from sklearn.externals import joblib
# joblib.dump(clf,'Chi2_income_predict.m')
scores_chi2 = cross_val_score(clf, Xt_chi2, y ,scoring='accuracy')
scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
print('CHI2--Accuracy:{}%'.format(np.mean(scores_chi2)*100))
print('CPEARSON--Accuracy:{}%'.format(np.mean(scores_pearson)*100))