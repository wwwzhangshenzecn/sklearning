from operator import itemgetter
import pandas as pd
import os,sys
from collections import defaultdict

# 加载打分表
data_filename = os.path.join('Data', 'ml-100k', 'u.data')
# userId,movieId,rating,timestamp
all_ratings = pd.read_csv(data_filename, delimiter='\t', names =['UserID', 'MovieID', 'Rating', 'DateTime'])
all_ratings["DateTime"] = pd.to_datetime(all_ratings['DateTime'], unit='s') #解析时间戳

# 加载电影表
movie_name_filename = os.path.join('Data', 'ml-100k', 'u.item')

movie_name_data = pd.read_csv(movie_name_filename,delimiter='|',encoding='mac-roman',
                              header=None, names =['MovieID', 'Title','release date','video release date','IMDb URL','unknown',
                                                   'Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama',
                                                   'Fantasy','Film-Noir ','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller',
                                                   'War','Western'])

print(movie_name_data[:5])
def get_movie_name(movie_id):
    title = movie_name_data.ix[int(movie_id)-1][1]
    # title_object = movie_name_data[movie_name_filename["MovieID"] == movie_id]["Title"]
    # title = title_object.values[0]
    return title

#Apriori算法的实现

# 新特征：Favorable 用于喜欢此电影 True/False type:bool
all_ratings['Favorable'] = all_ratings['Rating'] > 3

ratings = all_ratings[~all_ratings['UserID'].isin(range(200))]
all_favorable_ratings = ratings[ratings['Favorable']]#不处理不喜欢的
all_favorable_reviews_by_users = dict(
    (k, frozenset(v.values)) for k, v in all_favorable_ratings.groupby('UserID')['MovieID']
    )

ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
favorable_ratings = ratings[ratings['Favorable']] #进行表示喜欢的数据行的提取

# 创建用户打分过的电影聚集 dic{ userid: mives的集合 } type:str ,forzenset
favorable_reviews_by_users = dict(
    (k, frozenset(v.values)) for k, v in favorable_ratings.groupby('UserID')['MovieID']
    )

# 每部电影的影迷数量
num_favorable_by_movie = ratings[['MovieID', 'Favorable']].groupby('MovieID').sum()
#print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])

frequent_itemsets = {} # 项集
min_support = 50
# 喜爱次数大于最小支持度 的电影 {movieid}：Favorable ------ type: set:int
frequent_itemsets[1] = dict(
    (frozenset((movie_id, )), row['Favorable']) for movie_id, row in num_favorable_by_movie.iterrows() if row['Favorable'] > min_support
)
print("There are {} movies with more than {} favorable reviews".format(len(frequent_itemsets[1]), min_support))
print(frequent_itemsets)


# 在传入的相应项集中进行新规则的查找
def find_frequent_itemsets(favorable_reviews_by_users, k_l_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_l_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_support = itemset | frozenset((other_reviewed_movie, ))
                    counts[current_support] += 1

    return dict([(itemset, frequent) for itemset, frequent in counts.items() if frequent > min_support])
# 规则的发现
for k in range(2,20):
    cur_frequent_itemsets = find_frequent_itemsets(
        favorable_reviews_by_users, frequent_itemsets[k - 1], min_support
    )
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:

        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        # print(cur_frequent_itemsets)
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets

del frequent_itemsets[1]

# 便利每一部电影，作为结论，项集中的其他作为前提，组成一条规则
candidate_rules = []
for itemset_lenght, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise,conclusion))

print(candidate_rules[:200])

with open('movie_recommendator_model.txt','w',encoding='utf-8') as output_file:
    for k, v in candidate_rules:
        set_values = ' '.join([value for value in k])
        output_file.write(str(set_values)+'\t'+str(v)+'\n')

# 准确lu的判断
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user ,review in all_favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise ,conclusion = candidate_rule
        if premise.issubset(review):
            if conclusion in (review):
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

rule_confidences = { #计算准确率
    candidate_rule:correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])  for candidate_rule in candidate_rules
}

sort_confidence = sorted(rule_confidences.items(), key=itemgetter(1), reverse=True)
for index in range(100):
    print('Rule #{0}'.format(index + 1))
    (premise ,conclusion) = sort_confidence[index][0]
    premise_names = '  _and_  '.join(get_movie_name(idx) for idx in premise)
    print('Rule: If a person recommends {0} they will also recommend {1}'.format(premise_names, get_movie_name(conclusion)))
    print('- Confidence:{0:0.3f} %'.format(rule_confidences[(premise, conclusion)]*100))
    print("")