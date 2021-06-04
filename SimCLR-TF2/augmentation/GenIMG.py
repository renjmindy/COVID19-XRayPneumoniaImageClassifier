import numpy as np
import cv2 as cv

import tensorflow as tf
from keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.utils import data_utils

from GenUtil import *
import random

class GenIMG(data_utils.Sequence):
  def __init__(self, df, batch_size=16, sub='train', shuffle=True, info={}, width=80, height=80, VGG=False):

    super().__init__()
    self.df = df
    self.batch_size = batch_size
    self.sub = sub
    self.shuffle = shuffle
    self.info = info
    self.width = width 
    self.height = height
    self.VGG = VGG
    self.end()

  def __len__(self):
    return int(np.ceil(len(self.df) / float(self.batch_size)))

  def end(self):
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __retrieve__(self, index):

    X = np.empty((2 * self.batch_size, 1, self.height, self.width, 3), dtype=np.float32)
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    images = self.df.iloc[indexes]

    shuffle_c1 = np.arange(self.batch_size)
    shuffle_c2 = np.arange(self.batch_size)   

    if self.sub == 'train':
      random.shuffle(np.arange(self.batch_size))
      random.shuffle(np.arange(self.batch_size))
    else:
      random.seed(42)
      random.shuffle(np.arange(self.batch_size))
      random.shuffle(np.arange(self.batch_size))

      labels_c1 = np.zeros((self.batch_size, 2*self.batch_size))
      labels_c2 = np.zeros((self.batch_size, 2*self.batch_size))
      
      for i, j in enumerate(images.iterrows()):
        
        fn = j[1]['fn']
        self.info[index * self.batch_sieze + 1] = fn
        img = cv.cvtColor(cv.imread(fn), cv.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(np.array((img/255))).astype('float32')
        img_T1 = preprocess_for_train(img, self.height, self.width, color_distort=True, crop=False, flip=False, blur=False)
        img_T2 = preprocess_for_train(img, self.height, self.width, color_distort=True, crop=False, flip=False, blur=False)

        if self.VGG:
          img_T1 = tf.dtypes.cast(img_T1 * 255, tf.int32)
          img_T2 = tf.dtypes.cast(img_T2 * 255, tf.int32)
          img_T1 = preprocess_input(np.asarray(img_T1))
          img_T2 = preprocess_input(np.asarray(img_T2))

        X[shuffle_c1[i]] = img_T1
        X[self.batch_size+shuffle_c2[i]] = img_T2

        labels_c1[shuffle_c1[i], shuffle_c2[i]] = 1
        labels_c2[shuffle_c2[i], shuffle_c1[i]] = 1

      y = tf.concat([labels_c1, labels_c2], 1)
    
      return list(X), y, [None]
