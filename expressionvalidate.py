import tensorflow as tf
import numpy as np
import argparse

import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import preprocessing
from scipy.optimize import brentq
from scipy import interpolate
from keras.models import model_from_json
import keras
from keras.engine import Model
import cv2
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

def string_to_float(str):
    return float(str)

def evaluate(weight):
    model = model_from_json(
        open('./branchc51.json', 'r').read())
    print(model)
    model.load_weights(weight)

    fc8_layer = Model(inputs=model.input, outputs=model.get_layer(name='emo_fc8').output)

    correct = 0
    imgs_num = 0

    for root, dirs, files in os.walk('./validatedata-exp'):

        for d in dirs:
            dirpath = os.path.join(root, d)
            imgs = os.listdir(dirpath)
            imgnum = len(imgs)
            imgs_num += imgnum

            for j in range(imgnum):
                imgpath = os.path.join(dirpath, imgs[j])
                raw_img = cv2.imread(imgpath)
                test_img = cv2.resize(raw_img, (224, 224))
                test_img = np.array(test_img)
                test_img = (test_img.astype(np.float32)) / 255.0
                test_img = test_img - [[[0.3310429315873352, 0.2596515072870283, 0.21203281118510672]]]  # 这是身份的通道均值。
                test_img = np.expand_dims(test_img, axis=0)

                softmax_res = fc8_layer.predict(test_img, batch_size=1)
                max_index = np.argmax(softmax_res)
                if int(d) == max_index:
                    correct += 1

    acc = correct / imgs_num
    return acc

if __name__ == '__main__':
    weight_path = './result.hdf5'
    acc1 = evaluate(weight_path)
    print(acc1)