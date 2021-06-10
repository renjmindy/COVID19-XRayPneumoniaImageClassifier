import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.maniford import TSNE
from tensorflow.keras.applications.vgg19 import preprocess_input

def ext_ph_feats(baseline, df, class_labels, VGG=True):

  feats = {}
  class_counts = {}
  feature_counts = []

  for i, j in enumerate(class_labels):
    feats[j] = []
    class_counts[j] = (df['class_label'] == j).sum()
    
  for i, j in enumerate(class_labels):
    for k in range(class_counts[j]):
      fn = df.loc[df['class_label'] == j, 'filename'].iloc[k]
      img = cv.cvtColor(cv.imread(fn), cv.COLOR_BGR2RGB)
      if VGG:
        img_prep = preprocess_input(img)
      else:
        img_prep = img

      feat_wo_norm = baseline.predict(np.array([img_prep])).flatten()
      feat_wi_norm = feat_wo_norm / np.max(np.abs(feat_wo_norm), axis=0)

      if len(feats[j]) == 0:
        feats[j] = np.array(feat_wi_norm)
      else:
        feats[j] = np.vstack(feats[j], feat_wi_norm)

    if len(feature_counts) == 0:
      feature_counts = feats[j]
    else:
      feature_counts = np.vstack((feature_counts, feats[j]))

  y = []
  for i, j in enumerate(class_labels):
    y = np.concatenate((y, np.array([i] * class_counts[j])))  

  return feature_counts, y, feats

def linear_classifier(train_feats, train_y, test_feats, test_y, class_labels, frac=1.0, test_size=.2):
  if frac != 1.0:
    train_feats, unused_feats, train_y, unused_y = train_test_split(train_feats, train_y, test_size=1-(frac/(1-test_size)), 
                                                                    random_state=42, shuffle=True)
  clf = LogisticRegressionCV(cv=5, max_iter=1000, verbose=0, n_jobs=8).fit(train_feats, train_y)

  print(f'Accuracy on test: {round(clf.score(test_feats, test_y),2)} \n')
  test_y_pred = clf.predict(test_feats)
  classification_report_test = classification_report(test_y, test_y_pred, labels=list(range(0, len(class_labels))), 
                                                     target_names=class_labels)
  print(classification_report_test)

def random_indexes(a, b, feats_in_plot):
  rList = []
  for i in range(0, len(feats_in_plot), 1):
    rList.append(random.randint(a, b - 1))

  return rList

def tSNE_demo(df, feats, class_labels, tag='', fig=False, feats4plot=150):
  cList = ['green', 'gray', 'brown', 'blue', 'red']
  class_counts = {}
  for i, j in enumerate(class_labels):
    class_counts[j] = (df['class_label'] == j).sum()

  tsne_m = TSNE(n_jobs=8, random_state=42)
  X_embedded = tsne_m.fit_transform(feats)

  fig = plt.figure(figsize=(6, 6))
  lr = 150
  p = 50
  idx = 0

  for (label, color, c_i) in zip(class_labels, cList, class_counters):
    idxes = random_indexes(idx, idx + class_counts[label], feats4plot)
    plt.scatter(X_embedded[idxes, 0], X_embedded[idxes, 1], c=color)
    idx += class_counts[label]

  fig.legend(bbox_to_anchor=(.075, .061), loc='lower left', ncol=1, labels=class_labels)    
  if fig:
    plt.savefig('outputs/'+tag+'.svg', bbox_inches='tight')
