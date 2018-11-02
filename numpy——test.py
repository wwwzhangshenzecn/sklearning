# import numpy as np
#
# # a= np.zeros((2, 2, 3),dtype=np.uint8)
# # print(a.shape)
# # print(a)
# # a[0][1][1] = 2
# # print(a)
# '''
# # array([[[ 0,  1,  2,  3],
# #         [ 4,  5,  6,  7],
# #         [ 8,  9, 10, 11]],
# #
# #        [[12, 13, 14, 15],
# #         [16, 17, 18, 19],
# #         [20, 21, 22, 23]]])
# # '''
# # a = np.arange(24).reshape((2, 3, 4))
# # d = a[:,:,1]
# # print('d = a[:,:,1]\n',d)
# #
# # e = a[..., 1]
#
# l0 = np.arange(6).reshape((2, 3))
# l1 = np.arange(6, 12).reshape((2, 3))
# '''
# vstack 是指沿着纵轴拼接两个 arrary， vertical
# hstack 是指沿着横轴拼接两个array，     horizontal
# 更广义的拼接实现是用concatenate实现，horizonta后的两句依次等于vstack和hstack
# stack不是拼接而是在输入arrary的基础上增加一个新的维度
# '''
# m = np.stack((l0, l1))
# p = np.hstack((l0, l1))
# q = np.concatenate((l0, l1))
# r = np.concatenate((l0, l1), axis=-1)
# s = np.stack((l0, l1))
# print('l0 = np.arange(6).reshape((2, 3))\n',l0)
# print('l1 = np.arange(6, 12).reshape((2, 3))\n',l1)
#
# print('m = np.stack((l0, l1))\n',m)
# print('p = np.hstack((l0, l1))\n',p)
# print('q = np.concatenate((l0, l1))\n',q)
# print('r = np.concatenate((l0, l1), axis=-1)\n',r)
# print('s = np.stack((l0, l1))\n',s)
#
# import numpy.random as random
# print(random.rand(1, 3))
# print(random.sample((3, 3)))
# print(random.randint(1,10,10))
#
# import time
#
# n_test = 1000
# # 1000赌门，的中奖序列
# winning_doors = random.randint(1, 4, n_test)
# # 获胜次数
# winning = 0
# # 失败错误
# failing = 0
#
# for winning_door in winning_doors:
#     # 第一次尝试 猜测
#     first_try = random.randint(1, 4)
#     # 其他门的编号
#     remaining_choices = [i for i in range(1, 4) if i != first_try]
#     # 错误门的编号
#     wrong_doors = [i for i in range(1, 4) if i != winning_door]
#     # 从剩下的门中，去掉选的门
#     if first_try in wrong_doors:
#         wrong_doors.remove(first_try)
#
#     # 主持人打开一门
#     srceen_out = random.choice(wrong_doors)
#     remaining_choices.remove(srceen_out)
#     changed_mind_try = remaining_choices[0]
#
#     # 结果揭晓，记录下来
#     winning += 1 if changed_mind_try == winning_door else 0
#     failing += 1 if first_try == winning_door else 0
#
#
# print(
#     'You win {1} out of {0} tests if you changed your mind\n'
#     'You win {2} out of {0} tests if you insist on the initial choice'.format(
#         n_test, winning, failing
#     )
# )
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
# 通过rcParams 设置全局字体大小
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
np.random.seed(int(time.time()//10))

x = np.linspace(0, 5, 100)
y = 2 * np.sin(x)+ 0.3 * x**2
y_data = y + np.random.normal(scale=0.3, size=100)
plt.figure('data')
plt.plot(x, y_data, '.')

plt.figure('model')
plt.plot(x, y)

plt.figure('data & model')
plt.plot(x, y, 'k', lw=2)
plt.scatter(x, y_data)
plt.show()
