# coding=utf-8
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from operator import itemgetter

def func():
    df_train = pd.read_csv('csvFiles/tag_cooccurrence.csv')

    tags = df_train[df_train['id'].isin(range(20000))]['tags'].values
    # print(tags[:5])
    frequent_items = {}
    tag_dict = defaultdict(int)
    min_support = 50
    # 行程第一道规则
    for items in tags:
        tag_list = frozenset([v for v in  items.split(',')])
        for i, item_premise in enumerate(tag_list):
            for j, item_conclusion in enumerate(tag_list):
                if i == j: continue
                rule = frozenset([item_premise, item_conclusion])
                tag_dict[rule] += 1

    frequent_items[1] = dict(
        (k,v) for k, v in tag_dict.items() if v > min_support
    )

    def find_frequent_items(tags, k_l_items, min_support):
        counts = defaultdict(int )
        for reviews in tags: # 每一个id 后的标签集合
            reviews = frozenset(reviews.split(','))
            for item_pre in k_l_items: # 对每一条规则进行生产新的规则
                if item_pre.issubset(reviews):
                    for other_review_items in reviews - item_pre:
                        # set 不可哈希 元组可以
                        current_support = item_pre | frozenset((other_review_items,))
                        counts[current_support] += 1
                        # print(current_support,counts[current_support])

        return dict(
            [(itemset, frequent) for itemset, frequent in counts.items() if frequent > min_support]
        )

    for k in range(2,20):
        cur_frequent_itemsets = find_frequent_items(tags,frequent_items[k-1],min_support)
        frequent_items[k] = cur_frequent_itemsets
        if len(cur_frequent_itemsets) == 0 :
            print('Did not find any frequent itemsets of lenght{}'.format(k+1))
            sys.stdout.flush()
            break
        else:
            print('I find {} frequent itemsets of length {}'.format(len(cur_frequent_itemsets), k+1))
            sys.stdout.flush()
            frequent_items[k] = cur_frequent_itemsets

    # 将每一条符合的规则，去一个作为结论，其余的作为前提，形成最后的规则
    candidate_rules = []
    for itemset_lenght , items_counts in frequent_items.items():
        for itemset in items_counts.keys():
            itemset = set(itemset)
            for conclusion in itemset:
                premise = itemset - set((conclusion,))
                candidate_rules.append((frozenset(premise),conclusion))

    with open('csvFiles/tag_recommendator_model.txt', 'w', encoding='utf-8') as output_file:
        # 写入模型
        output_file.write('Premise,conclusion\n')
        for k, v in candidate_rules:
            set_values = ' '.join([value for value in k])
            output_file.write(str(set_values) + ',' + str(v) + '\n')

    correct_counts = defaultdict(int ) # 计数：规则正确/不正确
    incorrect_counts = defaultdict(int )

    tags_test = df_train[~df_train['id'].isin(range(20000))]['tags'].values
    for reviews in tags_test:
        reviews = frozenset(reviews.split(','))
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset( reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1
    # 准确率
    rules_confidence = {
        candidate_rule:correct_counts[candidate_rule] /float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule]+1) for candidate_rule in candidate_rules
    }
    sort_rules = sorted(rules_confidence.items(),key=itemgetter(1),reverse=True)

    for k in range(100):
        print('#{}'.format(k))
        (premise, conclusion) = sort_rules[k][0]
        # names = get_name_items(premise)
        print('if a person has {},they will have {}'.format(list(premise ), conclusion))
        print('-Confidence :{:.2f}%'.format(rules_confidence[(premise,conclusion)] * 100))
        print('')

    df_user_data = pd.read_csv('csvFiles/user_tag.csv')
    user_tags = defaultdict(int)
    tag_values = defaultdict(int)
    predict=defaultdict(int)

    for index, row in df_user_data.iterrows():

        tags = row[1].split(',')
        values = (row[2])[1:-1].split(',')
        for i in range(len(tags)):
            tag_values[tags[i]] = values[i]
        user_tags[index] = tag_values
        counts = defaultdict(int)
        tagsets = frozenset(tags)
        for premise,conclusion in candidate_rules:
            if premise.issubset(tagsets):
                    counts[conclusion] += 1

        counts = sorted(counts.items(), key=itemgetter(1),reverse=True)
        predict[index] = [tag for tag,count in counts[:10]]

    with open('csvFiles/user_recommendator.txt','w',encoding='utf-8') as file:
        file.write('user_id,predictor_tags\n')
        for index, tags in predict.items():
            tag = '"'+','.join(k for k in tags)+'"'
            file.write(str(index+1)+','+tag+'\n')





if __name__ == '__main__':
    # user_tags_predictor()
    func()