import numpy as np
import os
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.model_selection import cross_val_score

data_filename = os.path.join('Data','2014games.csv')
# 修复数据
results = pd.read_csv(data_filename,  skiprows= [0,])
results.columns = ['Date', 'Time', 'Visitor Team', 'VisitorPts', 'Home Team', 'HomePts', 'Score Type','Notes', 'OT?', 'ot']

#找出所有主场获胜的球队
results['HomeWin'] = results['VisitorPts'] < results['HomePts']
y_true = results['HomeWin'].values
print(results[:5])
print(y_true[:5])

#创建默认字典，存储球队上次比赛的结果
won_late = defaultdict(int)
results['HomeLastWon'] = False
results['VisitorLastWon'] = False

for index ,row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row['Visitor Team']
    #先简单处理一下
    row['HomeLastWin'] = won_late[home_team]
    row['BisitorLastWon'] = won_late[visitor_team]
    results.ix[index] = row
    #set current row
    won_late[home_team] = row['HomeWin']
    won_late[visitor_team] = not row['HomeWin']

print(results.ix[:5])

# clf = DecisionTreeClassifier(random_state=14)
# X_previouswins = results[['HomeLastWon','VisitorLastWon']].values
# print(X_previouswins[:5])
# scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
# print('Accuracy is {0:0.1f}%'.format(np.mean(scores) * 100))

ladder_filename = os.path.join('Data', '2013standings.csv')
ladder = pd.read_csv(ladder_filename , )
print(ladder[:5])

results['HomeTeamRankHigher'] = 0
for index ,row in results.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"

    home_rank = ladder[ladder['Team'] == home_team]['Rk'].values[0]
    visitor_rank = ladder[ladder['Team'] == visitor_team]['Rk'].values[0]
    row['HomeTeamRankHigher'] = int(home_rank > visitor_rank)
    results.ix[index] =row

print(results[:5])

clf = DecisionTreeClassifier(random_state=14)
X_previouswins = results[['HomeLastWon', 'VisitorLastWon', 'HomeTeamRankHigher']].values
# print(X_previouswins[:5])
# scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
# print('Accuracy is {0:0.1f}%'.format(np.mean(scores) * 100))

from sklearn.preprocessing import LabelEncoder # 用LabelEncoder转换器就能把字符型的球队名转换为整型
encoding = LabelEncoder()
encoding.fit(results['Home Team'].values) #将球队名转化为整型
home_team = encoding.fit_transform(results['Home Team'].values) # 抽取所有比赛的主客场的球队名（以转化为整型），并将其组合成一个矩形
visitor_team = encoding.fit_transform(results['Visitor Team'].values)
X_teams = np.vstack([home_team,visitor_team]).T
#决策树可以用这些特征值进行训练，但是DecisionTreeClassifiter仍把他们当成连续型特征。例如0-16的17支球队，算法会认为1和2相似，4和10不同。但是
#其实这意义，他们要么是一个球队要么不同，没有中间态
#为了消除这种和实际情况不一致的现象，我们可以使用OneHotEncode 转换器把这些整数转换为二进制数字（数据集可能会变得很大）

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
# 进行预处理和训练操作，结果保存备用
X_teams_expanded = onehot.fit_transform(X_teams).todense()
# clf = DecisionTreeClassifier(random_state=14)
# scores = cross_val_score(clf, X_teams_expanded ,y_true ,scoring='accuracy')
# print('Accuracy is {0:0.1f}%'.format(np.mean(scores) * 100))


from sklearn.ensemble import RandomForestClassifier # 随机森林估计器
# clf = RandomForestClassifier(random_state=14)
# scores = cross_val_score(clf, X_teams_expanded, y_true ,scoring='accuracy')
# print('Two features :Accuracy is {0:0.1f}%'.format(np.mean(scores) * 100))


X_home_rank = results[['HomeTeamRankHigher','HomeLastWon']].values
X_all = np.hstack([X_home_rank,X_teams])
# clf = RandomForestClassifier(random_state=14)
# scores = cross_val_score(clf, X_all, y_true ,scoring='accuracy')
# print('four features :Accuracy is {0:.1f}%'.format(np.mean(scores) * 100))

from sklearn.model_selection import GridSearchCV
parameter_space = {
                   "max_features": [2, 10, 'auto'],
                   "n_estimators": [100,],
                   "criterion": ["gini", "entropy"],
                   "min_samples_leaf": [2, 4, 6],
                   }
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print('GrifSearchCV :Accuracy is {0:.1f} %'.format(grid.best_score_ * 100))
