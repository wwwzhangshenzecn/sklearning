import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as tf
from PIL import Image, ImageFont, ImageDraw
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
def create_captcha(text, shear=0, size=(100, 24)):
    '''
    生成文字为text 的黑白图像 默认：100*24
    :param test: 图像文字
    :param shear: 错切指
    :param size: 图像大小
    :return: image（归一化处理）
    '''
    im = Image.new('L', size, color='black')
    draw = ImageDraw.Draw(im)
    # 设置字体
    draw.text((2,2), text, fill=1, font=ImageFont.truetype(r'Coval-Black.otf', 18))
    # 错切变化
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()



from skimage.measure import regionprops,label
def segement_iamge(image):
    '''
    将图像进行错切，然后进行分割
    :param image: 图像
    :return: 图个图像的列表
    '''
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y ,end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image, ]
    return subimages


image = create_captcha('LHLH', shear=0)

subimages = segement_iamge(image)
plt.imshow(subimages[0])

# 搭建神经网络
# from pybrain.tools.shortcuts import buildNetwork
# net = buildNetwork(400, 400, 26 ,bias=True)
from pybrain.supervised import BackpropTrainer
# trainer = BackpropTrainer(net)
import pickle
trainer = pickle.load(open('CAPTCHA_predictor.model','rb'))

# 创造一个图像-ABCD
image = create_captcha('ABDH', shear=0.1)
subimages = segement_iamge(image)

f, axes = plt.subplots(1,len(subimages), figsize=(10, 3))


from pybrain.datasets import SupervisedDataSet
testing = SupervisedDataSet(400,1)
for i in range(len(subimages)):
    testing.addSample(resize(subimages[i],(20, 20)).flatten(), i)

predictions = trainer.testOnClassData(dataset=testing)
for k, v in enumerate(predictions):
    print(letters[v],end='')

print()


