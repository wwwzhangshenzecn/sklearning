# -*-coding:utf-8-*-
'''
数字图片识别
'''
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform as tf
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw, ImageFont
from pybrain.datasets import SupervisedDataSet
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import os,time
import warnings
import pickle
warnings.filterwarnings('ignore')

if not os.path.exists('Number_png'):
    os.mkdir('Number_png')

numbers = [1,2,3,4,5,6,7,8,9,0]
shears = [0.1,0.2,0.3,0.4,0,-0.1,-0.2,-0.3-0.4]
def create_number(text, shear=0, size=(20,20)):
    '''
    使用L模式创建一张空白，设置字体等进行斜切效果，归一化处理返回image
    :param text: 图片文本
    :param shear: 斜切度
    :param size: 图片大小
    :return:
    '''
    im = Image.new('L', size, color='black')
    draw = ImageDraw.Draw(im)
    draw.text(xy=(6, 1),text=text, fill=1, font=ImageFont.truetype(r'Coval-Black.otf', 15))
    image = np.array(im)
    image = tf.warp(image, tf.AffineTransform(shear=shear))

    seg = resize(segement_image(image)[0],(20, 20))

    name = text+'-'+str(time.time())+'.png'
    path = os.path.join('Number_png',name)
    plt.imsave(path ,seg)
    return image / image.max() # 归一化处理

def segement_image(image):
    label_iamge = label(image > 0)
    subimages = []
    for region in regionprops(label_iamge):
        start_x, start_y ,end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image, ]
    return subimages

from sklearn.utils import check_random_state
sample_rate=check_random_state(int(time.time())%10)
def generate_sample(random_rate=None):
    '''
    生成样本图片
    :param random_rate:
    :param shear: 斜切
    :return:
    '''
    random_rate = check_random_state(random_rate)
    shear = random_rate.choice(shears)
    number = random_rate.choice(numbers)
    return create_number(str(number), shear=shear), numbers.index(number)
image, target = generate_sample(sample_rate)

# 创建样例
datasets , targets = zip(*(generate_sample(sample_rate) for i in range(10000)))
datasets = np.array(datasets ,dtype='float') # 数据变为np。array类型
targets = np.array(targets)

oneHot = OneHotEncoder() # 数据归一化，一个矩阵形式，shape[0] * 标签数
y = oneHot.fit_transform(targets.reshape(targets.shape[0], 1)).todense()
datasets = np.array([resize(segement_image(sample)[0], (20, 20)) for sample in datasets])

# 将数据扁平化
X = datasets.reshape((datasets.shape[0], datasets.shape[1] * datasets.shape[2]))
X_train ,X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

# 添加数据到数据格式中
training =SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])
testing = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])

# 搭建三层网络
net = buildNetwork(X.shape[1], 150, y.shape[1] ,bias=True)
# 使用BP算法
trainer = BackpropTrainer(net, training ,weightdecay=0.01)
# 训练步数
trainer.trainEpochs(epochs=50)
# 保存模型
# model_filename = open('CAPTCHA_predictor.model','wb')
# pickle.dump(trainer,model_filename,0)
# model_filename.close()

predictions = trainer.testOnClassData(dataset=testing)

from sklearn.metrics import f1_score,classification_report
print(classification_report(y_test.argmax(axis=1), predictions))


