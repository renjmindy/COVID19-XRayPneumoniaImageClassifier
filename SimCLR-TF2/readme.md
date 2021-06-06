# Implementation of Contrastive Learning from A Simple Self-supervised Algorithm 

   Fig. 1                     |   Fig. 2
:----------------------------:|:------------------------------:
![](./imgs/demo_simclr_1.png) | ![](./imgs/demo_simclr_2.gif)

## Differences among Supervised, Unsupervised, Semi-supervised and Self-supervised

  Supervised | Unsupervised | Semi-supervised | Self-supervised 
:-----------:|:------------:|:---------------:|:---------------:
trained w/ labeled input | unlabeled input | labeled input | 
labeled predictions | no output | unlabeled predictions | 
regression, classification w/ known patterns | clustering w/ unknown patterns | w/ weakly known patterns  |


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
