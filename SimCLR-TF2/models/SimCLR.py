from datetime import datetime
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from keras.utils.layer_utils import count_params
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from SoftmaxCosineSim import SoftmaxCosineSim
from SimCLS import GenClassifier as modelgen
from swish import swish

class SimCLR:
  def __init__(self, baseline, imgshape, batch_size, feat_dim, feat_dims_ph, num_of_unfrozen_layers, lr_dense=.005, lr_class=.0001, 
               loss='categorical_crossentropy', output='outputs/trashnet', r=1):
    self.baseline = baseline
    self.imgshape = imgshape
    self.batch_size = batch_size
    self.feat_dim = feat_dim # feature dimensions 
    self.feat_dims_ph = feat_dims_ph # feature dimensions (projection head)
    self.num_layers_ph = len(feat_dims_ph) 
    self.num_of_unforzen_layers = num_of_unfrozen_layers
    self.lr_dense = lr_dense
    self.lr_class = lr_class
    self.optimizer = Adam(lr_class, amsgrad=True)
    self.loss = loss
    self.output = output
    self.r = r

    self.flatten_layer = Flatten()
    self.soft_cos_sim = SoftmaxCosineSim(batch_size=self.batch_size, feat_dim=self.feat_dim)

    self.phList = []
    for i in range(0, self.num_layers_ph, 1):
      if i < self.num_layers_ph - 1:
        self.phList.append(Dense(feat_dims_ph[i], activation='swish', kernel_regularizer=l1(lr_dense)))
      else:
        self.phList.append(Dense(feat_dims_ph[i], kernel_regularizer=l1(lr_dense)))
        
    self.SimCLR_classifier = self.build_classifier()

  def build_classifier(self):
    for layer in range(0, len(self.baseline.layers), 1):
      if layer < self.num_of_unfrozen_layers:
        self.baseline.layers[layer].trainable = False
      else:
        self.baseline.layers[layer].trainable = True

    iList = [] # input images
    fList = [] # baseline output
    hList = [] # base encoder
    gList = [] # projection head

    for i in range(0, self.num_layers_ph, 1):
      gList.append([])      

    baseline = self.baseline
    phList = []
    for i in range(0, self.num_layers_ph, 1):
      phList.append(self.phList[i])

    for idx in range(0, 2 * self.batch_size, 1):
      iList.append(Input(shape=self.imgshape))
      fList.append(baseline(iList[idx]))
      hList.append(self.flatten_layer(fList[idx]))
      for i in range(0, self.num_layers_ph, 1):
        if i == 0:
          gList[i].append(phList[i](hList[idx]))
        else:
          gList[i].append(phList[i](gList[i-1][idx]))

    ph_output = self.soft_cos_sim(gList[-1])

    SimCLR_classifier = Model(inputs=iList, outputs=ph_output)
    SimCLR_classifier.compile(optimizer=self.optimizer, loss=self.loss)

    return SimCLR_classifier

    def build_trainer(self, train_datagen, val_datagen, epochs=10, prout=True):
      checkpoint, earlyStopping, reduce_lr = self.get_callbacks()
      SimCLR_classifier = self.SimCLR_classifier
      SimCLR_classifier.fit(train_datagen, epochs=epochs, verbose=1, validation_data=val_datagen, callbacks=[checkpoint, earlyStopping, reduce_lr])

      if prout:
        self.print_weights()

      self.saveBaseline()

    def build_defreezer(self, train_datagen, val_datagen, num_of_unfrozen_layers, r, lr_class=.0001, epochs=10, prout=True):

      self.num_of_unfrozen_layers = num_of_unfrozen_layers
      self.r = r
      if self.lr_class != lr_class:
        self.change_lr(lr_class)

      self.SimCLR_classifier = self.build_classifier()

      if prout:
        self.print_weights()

      self.build_trainer(train_datagen, val_datagen, epochs)

    def build_predicter(self, data):
      return self.SimCLR_classifier.predict(data)

    def saveBaseline(self):
      self.baseline.save(self.output+'/baseline_round_'+str(self.r)+'.h5')

    def change_lr(self, lr_class):
      self.lr_class = lr_class
      K.set_value(self.SimCLR_classifier.optimizer.learning_rate, self.lr_class)

    def get_callbacks(self):
      current = datetime.now()
      current_string = current.strgtime('_%m_%d_%Hh_%M')

      checkpoint = ModelCheckpoint(self.output+'/SimCLR'+current_string+'.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto')   
      earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=True)
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=.5)

      return checkpoint, earlyStopping, reduce_lr

    def print_weights(self):
      no_trainable = count_params(self.SimCLR_classifier.trainable_weights)
      no_nontrainable = count_params(self.SimCLR_classifier.non_trainable_weights)   

      print(f'trainable counts (1): {round(no_trainable/1e6, 2)} M.')
      print(f'trainable counts (2): {len(self.SimCLR_classifier.trainable_weights)}')
      print(f'non-trainable counts (1): {round(no_nontrainable/1e6, 2)} M.')
      print(f'non-trainable counts (2): {len(self.SimCLR_classifier.non_trainable_weights)}')

    def build_tester(self, df, batch_size, params, frac, class_labels, lr_dense=.005, lr_class=.005, nums_of_unfrozen_layers=[5,5,6,7],
                     lrs=[1e-3, 1e-4, 5e-5, 1e-5], epochs=[5, 5, 20, 25], verbose_epoch=0, verbose_cycle=1):
      results = {'acc':0}
      for i in range(0, 5, 1):
        if verbose_cycle:
          print(f'Learning attempt {i+1}')

        classifier = modelgen(baseline=self.baseline, class_labels=params['num_classes'], lr_dense=lr_dense, lr_class=lr_class)
        train_datagen, val_datagen, test_datagen = classifier.datagenerator(df, frac, batch_size, params)
        classifier.datatrainer(train_datagen, val_datagen, frac, nums_of_unfrozen_layers, lrs, epochs, verbose_epoch, verbose_cycle)
        acc, report = classifier.datatester(df['test'], test_datagen, class_labels)

        if results['acc'] < acc:
          results['acc'] = acc
          results['report'] = report
          results['attempt'] = i+1

        print(f'Best result from attempt {results['attempt']}')
        print(results['report'])
