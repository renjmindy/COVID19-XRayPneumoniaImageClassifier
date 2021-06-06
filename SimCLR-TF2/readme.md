# Implementation of A Simple Self-supervised Algorithm for Contrastive Learning in Visual Representations 

   Fig. 1                     |   Fig. 2
:----------------------------:|:------------------------------:
![](./imgs/demo_simclr_1.png) | ![](./imgs/demo_simclr_2.gif)

## Differences among Supervised, Unsupervised, Semi-supervised and Self-supervised

  Supervised | Unsupervised | Semi-supervised | Self-supervised 
:-----------:|:------------:|:---------------:|:---------------:
pure (1) | pure (2) | hybrid of (1) and (2) | sub of (2)
train labeled input | train unlabeled input | train labeled input | train unlabeled input
predict labeled input | no predictions | predict unlabeled input | predict unlabeled input
test labeled input | no testing | test labeled input | test unlabeled input
regression | clustering | binary classification | segmentation
binary or multiple classification | association | binary or multiple classification


## Requirements:
  
  * jupyterlab==2.0.1
  * Keras==2.3.1
  * tensorflow==2.1
  * pandas==0.22.0
  * opencv-python==4.2.0.32
  * scikit-learn==0.23.1
  * scipy==1.4.1
  * numpy==1.18.1
  * DateTime==4.3
