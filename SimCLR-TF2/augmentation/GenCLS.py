import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.utils import data_utils

class GenImgs(data_utils.Sequence):
  def __init__(self, df, batch_size=16, sub='train', shuffle=True, prep=None, info={}, maxW=80, maxH=80, multi_classes=5, 
               VGG=True, aug=None):
    
    super().__init__()
    self.indexes = np.asarray(self.df.index)
    self.shuffle = shuffle
    self.sub = sub
    self.batch_size = batch_size
    self.prep = prep
    self.info = info
    self.maxW = maxW
    self.maxH = maxH
    self.multi_classes = multi_classes
    self.VGG = VGG
    self.aug = aug
    self.gen = gen
    self.end()

  def __len__(self):
    return int(np.ceil(len(self.df) / float(self.batch_size)))

  def end(self):
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def gen(self):
    return ImageDataGenerator

  def __retrieve__(self, index):
    
    X = np.empty((self.batch_size, self.maxH, self.maxW, 3), dtype=np.float32)
    y = np.empty((self.batch_size, self.multi_classes), dtype=np.float32) 
    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    images = self.df.iloc[indexes]

    for i, j in enumerate(images.iterrows()):
      fn = j[1]['filename']
      self.info[index * self.batch_size + 1] = fn
      img = cv.cvtColor(cv.imread(fn), cv.COLOR_BGR2RGB)
      
      if self.VGG:
        X[i, ] = preprocess_input(np.asarray(img))
      else:
        X[i, ] = img

      if self.sub == 'train':
        y[i, ] = i[1]['class_one_hot']

    if self.prep not None:
      X = self.prep(X)

    return X, y, [None]
