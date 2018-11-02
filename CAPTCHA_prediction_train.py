# conding=utf-8
'''
把大图像分成只包含一个字母的4张小图像
为每一个字母分类
把字母重新组合
用词典修正单词识别错误
'''
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
def create_captcah(text, shear = 0, size=(100, 24)):
    # 生成黑白图像
    im = Image.new('L',size,color='black')
    draw = ImageDraw.Draw(im)
    # 设置字体等
    font = ImageFont.truetype(r'Coval-Black.otf',18)
    draw.text((2, 2), text, fill=1, font=font)

    # 错切变化效果
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear = shear)
    image = tf.warp(image, affine_tf)
    return image / image.max() # 归一化处理


image = create_captcah('ABSL' ,shear=0.5)
plt.imshow(image, cmap='Greys')
plt.show()

# 将图像且分为单个字母
from skimage.measure import label, regionprops
def segment_image(image):
    labeled_image = label(image )
    subimages = [] # 切割小图像存在
    regionprop = regionprops(labeled_image)
    if len(regionprop) > 4:
        regionprop = regionprop[:-1]
    for region in regionprop:
        start_x, start_y ,end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])

    if len(subimages) == 0:
        return [image,]
    return subimages

subimages = segment_image(image)
f, axes = plt.subplots(1,len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i],cmap='Greys')
plt.show()


# 创建数据集
from sklearn.utils import check_random_state
random_state = check_random_state(14)
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
shear_values = np.arange(0, 0.5, 0.05)

def generate_smaple(random_state = None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    # print(letter)
    return create_captcah(letter, shear=shear, size=(20, 20)),letters.index(letter)

image, target = generate_smaple(random_state)
plt.imshow(image, cmap='Greys')
plt.show()
print('The target for this image is :{0}'.format(target))

# 生成3000个字母的样本空间
datsets, targets = zip(*(generate_smaple(random_state)for i in range(3000)))
datset = np.array(datsets, dtype='float')
targets = np.array(targets)

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))
# 将稀疏矩阵转化为密集矩阵
y = y.todense()

from skimage.transform import resize # 重新调整图像的大小，方便处理
datset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in datset])

# 创建数据集，datsets数组是三维的，因为他里面是存储的二维图像信息。由于分类器接受的是二维数组，需要将此扁平化
X = datset.reshape((datset.shape[0], datset.shape[1] * datset.shape[2]))
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

# 训练和分类
# 数据集是每一张20*20的黑白图像，徒有400个特征，输入神经网络，输出26和0和1之间的值，值越大，字母越大
# 使用PyBrain构建神经网络分类器
# SuperviseDataSet 一种数据集格式

from pybrain.datasets.supervised import SupervisedDataSet
training = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i],y_train[i])
testing = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i],y_test[i])

# 创建一个最基础的三层结构神经网络，输入层输出层和一层隐含层
# 输入层：每一张图20*20，具有400个特征，需要400个神经元
# 输出层：需要得到26个可能的类别，此时需要26个输出神经元
# 一层隐含层：若是神经元数量过多，神经网络呈现稀疏的特点，容易出现过拟合现象；若是数量过少，出现训练不足，发生低拟合现象
# 使用buildNetwork函数，指定维度，创建神经网络。
# 参数1： 输入层神经元数 ；2：隐藏层神经元数 ； 3：输出层神经元数

from pybrain.tools.shortcuts import buildNetwork
# print(X.shape[1],100,y.shape[1])
net = buildNetwork(X.shape[1], 150, y.shape[1], bias=True)
# backprop
from pybrain.supervised.trainers.backprop import BackpropTrainer
trainer = BackpropTrainer(net, training, learningrate = 0.01, weightdecay=0.01)
# 相比于迭代收敛，直接使用固定步数更简单直接
trainer.trainEpochs(epochs=20)

# 模型的保存
import pickle
model_filename = open('CAPTCHA_predictor.model','wb')
pickle.dump(trainer,model_filename,0)
model_filename.close()

predictions = trainer.testOnClassData(dataset=testing)




from sklearn.metrics import f1_score,classification_report
print(classification_report(y_test.argmax(axis=1), predictions))
# print("F-scores:{0.2f}".format(f1_score(predictions, y_test.argmax(axis=1))))

# 采用二维混淆矩阵来表示识别错误的字母
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test,axis=1), predictions)
plt.figure(figsize=(10, 10))
plt.imshow(cm)
tick_marks = np.arange(len(letters))
plt.xticks(tick_marks,letters)
plt.yticks(tick_marks,letters)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


def predict_captcha(captcha_image, net):
    # 分割图像
    subimages = segment_image(captcha_image)
    if len(subimages) == 3:
        f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
        for i in range(len(subimages)):
            axes[i].imshow(subimages[i], cmap='Greys')

        plt.show()

    predict_word = ''
    for subimage in subimages:
        subimage = resize(subimage, (20, 20))
        # 激活神经网络
        outputs = net.activate((subimage.flatten()))
        prediction = np.argmax(outputs)
        # print(prediction)
        predict_word += letters[prediction]

    return predict_word

word = 'GENE'
captcha = create_captcah(word, shear=0.2)
plt.imshow(captcha)
plt.show()
print(predict_captcha(captcha, net))


def test_prediction(word, net , shear=0.2):
    # print(word)
    captcha =create_captcah(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = prediction[:4]
    # print(word,prediction)
    return word == prediction, word, prediction

# 使用NLTK中words生成单词
from nltk.corpus import words
valid_words = [word.upper() for word in words.words() if len(word) == 4]
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = test_prediction(word, net, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1

print('Number correct is {}'.format(num_correct))
print('Number incorrect is {}'.format(num_incorrect))

# 使用字典增加预测，采用的是单词距离-长度减去不同字母的个数
from operator import itemgetter
def compute_distance(prediction, word):
    return len(prediction) - sum([prediction[i] == word[i] for i in range(len(prediction))])

def improved_prediction(word, net, dictionary, shear=0.2):
    captcha = create_captcah(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = prediction[:4]
    if prediction not in dictionary:
        distance = sorted([(word, compute_distance(prediction,word)) for word in dictionary], key=itemgetter(1))
        best_word = distance[0]
        prediction = best_word[0]

    return word == prediction, word, prediction

from nltk.corpus import words
valid_words = [word.upper() for word in words.words() if len(word) == 4]
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = improved_prediction(word, net, valid_words, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1

print('Number correct is {}'.format(num_correct))
print('Number incorrect is {}'.format(num_incorrect))

