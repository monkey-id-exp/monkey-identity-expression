import keras
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras import regularizers, initializers
import numpy as  np
import pandas as pd
import os
from keras.models import model_from_json

# 新加一个用于画出图 用tensorboard来画
keras.callbacks.TensorBoard(log_dir='./results/graph',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./results/graph',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

# custom parameters
nb_class1 = 4
nb_class2 = 60
hidden_dim = 512
img_w = 224
img_h = 224
batchsize = 64
epochs = 200
dropout_rate = 0.5
weight1 = 0.5
weight2 = 0.5

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool4').output
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_1')(last_layer)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_2')(x1)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_3')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='emo_pool5')(x1)
x1 = Flatten(name='flatten1')(x1)
x1 = Dense(hidden_dim, activation='relu', name='emo_fc6', kernel_regularizer=regularizers.l2(0.001))(x1)
x1 = Dense(hidden_dim, activation='relu', name='emo_fc7', kernel_regularizer=regularizers.l2(0.001))(x1)
x1 = Dropout(dropout_rate)(x1)
out1 = Dense(nb_class1, activation='softmax', name='emo_fc8', kernel_regularizer=regularizers.l2(0.001))(x1)

x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_1')(last_layer)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_2')(x2)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_3')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='id_pool5')(x2)
x2 = Flatten(name='flatten2')(x2)
x2 = Dense(hidden_dim, activation='relu', name='id_fc6',
           kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None))(
    x2)
x2 = Dense(hidden_dim, activation='relu', name='id_fc7',
           kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None))(
    x2)
x2 = Dropout(dropout_rate)(x2)
out2 = Dense(nb_class2, activation='softmax', name='id_fc8',
             kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',
                                                             seed=None))(x2)
custom_vgg_model = Model(vgg_model.input, [out1, out2])

for layer in custom_vgg_model.layers[:12]:
    layer.trainable = False
for layer in custom_vgg_model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)

custom_vgg_model.summary()
sgd = SGD(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto')

custom_vgg_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                         optimizer=sgd,
                         loss_weights=[weight1, weight2],
                         metrics=['accuracy'])

print('model is ready')

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_frame = pd.read_excel('train_id_exp.xls')

def encode(data_frame, col):
    col_lst = list(set(data_frame[col].values))
    col_value = [1 * (np.array(col_lst) == value) for value in data_frame[col]]
    data_frame[col] = col_value
    return data_frame

for data_col in ['id', 'emo']:
    train_frame = encode(train_frame, data_col)

train_frame.emo = np.load("exponehot.npy", allow_pickle=True)
train_frame.id = np.load("idonehot.npy", allow_pickle=True)

train_generator = train_datagen.flow_from_dataframe(
    train_frame,
    directory='./traindata',
    x_col='path',
    y_col=['emo', 'id'],
    target_size=(img_w, img_h),
    batch_size=batchsize,
    class_mode='multi_output')

N_train = train_generator.n
print("N_train:", N_train)

print('generator is ok,ready to train')

checkpointer = ModelCheckpoint('./results/model_{epoch:03d}.hdf5',
								verbose=0,
								save_best_only=False,
								save_weights_only=True,
								period=1)

custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=N_train / batchsize,
    epochs=epochs,
    callbacks=[tbCallBack,checkpointer])
print('training is over')

json_string = custom_vgg_model.to_json()
open('./results/branchc51.json', 'w').write(json_string)
print('save model successfully')
