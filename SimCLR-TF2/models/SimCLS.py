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
from tensorflow.keras.regularizers import ll
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
import GenCLS import GenClassifier as datagen
from swish import swish

class SimCLS:
  def __init__(self, baseline, multi_classes, batch_size=32, lr_dense=.005, lr_class=.005, output='models/'):

    self.baseline = baseline
    self.multi_classes = multi_classes
    self.batch_size = batch_size
    self.lr_dense = lr_dense
    self.lr_class = lr_class
    self.output = output
    self.classification = self.dataclassifier()

  def dataclassifier(self):
    baseline = self.baseline
    x = baseline.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='swish', kernel_regularizer=l1(self.lr_dense))(x)
    classification = Dense(self.multi_classes, activation='softmax', kernel_regularizer=l1(self.lr_class))(x)
    optimizer = Adam(lr=.001, amsgrad=True)

    classification = Model(baseline.input, classification)
    classification.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return classification

  def datagenerator(self, dfs, frac, batch_size, params):
    train_datagen = datagen(dfs['train'].sample(frac=frac).reset_index(drop=True), batch_size=batch_size, shuffle=True, **params)
    val_datagen = datagen(dfs['val'].reset_index(drop=True), batch_size=batch_size, shuffle=False, **params)
    test_datagen = datagen(dfs['test'].reset_index(drop=True), batch_size=1, shuffle=False, **params)

    return train_datagen, val_datagen, test_datagen

  def dataregularizer(self, frac):
    cp = ModelCheckPoint(self.output+ "/best_classifier_frac_"+ str(fraction)+ ".h5", monitor='val_loss', 
                                 verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=0, factor=.5)

    return cp, es, lr

  def datadefreezer(self, num_of_unfrozen_layers, pr=False):
    for layer in range(0, len(self.classification.layers), 1):
      if layer < number_of_unfrozen_layers:
        self.classification.layers[layer].trainable = False
      else:
        self.classification.layers[layer].trainable = True

    if pr:
      no_trainable = count_params(self.classification.trainable_weights)
      no_nontrainable = count_params(self.classification.non_trainable_weights)

      print(f'trainable counts (1): {round(no_trainable/1e6, 2)} M.')
      print(f'trainable counts (2): {len(self.classification.trainable_weights)}')
      print(f'non-trainable counts (1): {round(no_nontrainable/1e6, 2)} M.')
      print(f'non-trainable counts (2): {len(self.classification.non_trainable_weights)}')

  def modelfitter(self, train_datagen, val_datagen, test_datagen, frac, num_of_unfrozen_layers, lrs, epoches, 
                  verbose_epoches=0, verbose_cycles=1):
    classification = self.dataclassifier()
    checkpoint, earlystopping, reduce_lr = self.dataregulaizer(frac)

    for i, (num_of_unfrozen_layers, lr, ep) in enumerate(zip(num_of_unfrozen_layers, lrs, epoches)):
      self.datadefreezer(num_of_unfrozen_layers)
      classification = self.dataclassifier
      K.set_value(classification.optimizer.learning_rate, lr)

      history = classification.fit(train_datagen, epoches=ep, verbose=verbose_epoches, validation_data=val_datagen,
                                   callbacks=[earlystopping, reduce_lr]
                                   #callbacks=[earlystopping, checkpoint, reduce_lr])

      if verbose_cycle:
         print(f"CYCLE {i}: num_of_unfrozen_layers: {num_of_unfrozen_layers}" + f" - epochs: {ep} - lr: {lr:.1e}", end=" | ")
         print(f"Training Loss at end of cycle: {history.history['loss'][-1]:.2f}" + 
               f"- Training Acc: {np.max(history.history['categorical_accuracy']):.2f}" + 
               f"- Validation Acc: {np.max(history.history['val_categorical_accuracy']):.2f}")

      if np.isnan(history.history['val_loss']).any():
        print("Learning diverged, stopped.")
        break

  def modeltester(self, df_test, test_datagen, class_labels):
    predictions = self.dataclassifier.predict(test_datagen, steps=len(df_test.index))

    y_true_test = []
    y_pred_test = []

    for y_t, y_p in zip(test_datagen, predictions):
      y_true_test.append(np.argmax(y_t[1]))
      y_pred_test.append(np.argmax(y_p))

    acc = np.mean(np.equal(y_true_test, y_pred_test))
    if .3 < acc:
      classification_report_test = classification_report(y_true_test, y_pred_test, 
                                                         labels=list(range(0, len(class_labels))), target_names=class_labels)
    else:
      classification_report_test = None
                                   
    return acc, classification_report_test
