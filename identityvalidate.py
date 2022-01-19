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
from keras.engine import  Model
import cv2
from sklearn.model_selection import KFold

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
def get_paths(pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = pair[0] + '/' + pair[1]
            path1 = pair[0] + '/' + pair[2]
            issame = True
        elif len(pair) == 4:
            path0 = pair[0] + '/' + pair[1]
            path1 = pair[2] + '/' + pair[3]

            issame = False
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, issame_list
embeddings=[]
def getembedding(model,paths):
    global embeddings
    imgnum = len(paths)
    flag = 0
    a = True
    for i in range(imgnum):
        raw_img = cv2.imread(paths[i])
        test_img = cv2.resize(raw_img, (224, 224))
        test_img = np.array(test_img)
        test_img = (test_img.astype(np.float32)) / 255.0
        test_img = test_img - [[[0.3310429315873352, 0.2596515072870283, 0.21203281118510672]]]  # 这是身份的通道均值。
        test_img = np.expand_dims(test_img, axis=0)

        if flag==0 :
            embeddings = model.predict(test_img, batch_size=1)
            flag=1
        else:
            dp1_output = model.predict(test_img, batch_size=1)
            embeddings = np.append(embeddings, dp1_output,axis=0)
    return embeddings

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):

    assert (len(embeddings1) == len(embeddings2))
    nrof_pairs = min(len(actual_issame), len(embeddings1))

    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 2000, 10)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    return tpr, fpr, accuracy

def main():
    pairs = read_pairs('./identityvalidate.txt')
    paths, actual_issame = get_paths(pairs)
    ACC=[]
    model = model_from_json(open('./branchc51.json', 'r').read())
    weight_path = './result.hdf5'
    model.load_weights(weight_path)
    dp1_layer = Model(inputs=model.input, outputs=model.get_layer(name='dropout_2').output)
    dp1_layer.summary()

    embeddings_list = getembedding(dp1_layer,paths)
    tpr, fpr, acc = evaluate(embeddings_list, actual_issame)
    print('acc:', acc)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(acc), np.std(acc)))
    ACC.append(np.mean(acc))
    print(ACC)

if __name__ == '__main__':
    main()
