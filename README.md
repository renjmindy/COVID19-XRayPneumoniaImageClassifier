# Diagnosis of COVID-19 alike Viral Pneumonia: 
## Building CNN from Scratch for Pneumonia Diagnosis by Classifying Chest X-Ray Images

![simpsonXray](https://github.com/renjmindy/dsc-mod-4-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/maxresdefault1.jpg)

![simpsonXray](https://github.com/renjmindy/dsc-mod-4-project-v2-1-onl01-dtsc-ft-052620/blob/master/images/maxresdefault2.jpg)

## Motivation

The `computer vision` possesses a broad range of applications in use for a great variety of domains, one of which is the healthcare-related medical industry. Artificial intelligence is now transforming healthcare. Besides computer vision, AI healthcare companies make good use of traditional machine learning algorithms and natural language processing as the cutting-edge tool-kits with greatest potential so as to explore everything ranging from drug chemistry to generic markers. For customer services, these companies are offering online consultants with predictive analytics, and meanwhile, they are incorporating test results and sensor data gathered from current patients in order to provide more and more real-time status updates to medical practitioners. As mentioned previously, the most exciting areas for AI in healthcare, are around `computer vision` and `natural language processing`. Here, there are four vivid examples demonstrating how both of them benefit healthcare. (1) [cancer screening](https://www.youtube.com/watch?v=XLb0xUe80uo&feature=emb_title) with computer vision; (2) [facilitating both onsite and remote diagnosis]()

## Introduction

Now that I have possessed background knowledge regarding how CNNs work and how to build them using Keras, its time to make good use of those skills a little more independently in order to build a CNN from scratch on my own in order to solve an image recognition problem. In this project, I'll conduct building an image classifier from the most beginning through developing CNN models where overfitting issues are resolved by employing regularizations to the end.  

## Objectives

For this module's final project, I have the choice of the study:

- Image Classification with Deep Learning

I pick up one chest x-ray image classification problem as my project, and I plan to tackle it by means of deep learning, because I consider `Portfolio Depth`.

This chosen option will allow myself to practice all the necessary skills in a group of settings, before diving into my individual projects with an emphasis on the `computer vision`. I will produce a couple of capstone projects that are *not only* more polished and sophisticated, *but also* my portfolio will demonstrate even more breadth by **constructing various deep learning models that are in applicable use in order to greatly reduce both the time and labor consumption arrising from the workflow of diagnosis in hospitals**. A series of computer-vision-focused projects following after solving this image classification problem would be (1) [image caption generator](https://github.com/renjmindy/AutomaticImageCaptionGenerator) and (2) [image detectors](https://github.com/renjmindy/ImageDetectors) as well. They will be addressed in great details at links as indicated above.

Through this project, I will: 

- Load images from a hierarchical file structure using an image datagenerator 
- Apply data augmentation to image files before training a neural network 
- Build a CNN using Keras 
- Visualize and evaluate the performance of CNN models 
- Load saved Keras models 
- Use Keras methods to visualize activation functions in CNNs 
- Take advantage of pretrained networks
- Study how pre-trained neural networks benefit feature extraction 
- Understand what "freezing" and "unfreezing" a layer means in a neural network 
- Implement feature engineering and fine tuning on a pre-trained model 
- Use Keras to adapt a pretrained CNN 

## Loading Data for Image Classification with Deep Learning

The data for this project concerns lung xray images for pneumonia. The original dataset is from Kaggle. I have downloaded the entire dataset for the sake of model training in order to design various architectures and evaluate their performaces as well by fitting to data. 
⏰  
It is anticipated that this process will take approximately hours (even overnight) to run on a standard machine, although times will vary depending on every individual's particular computer and set up. For your reference, downsampled dataset can be found in a subdirectory, **data**, of this repository. 

To build a deep neural network that trains on a large dataset for classification is a non-trivial task. In this case, I utilize x-ray images of pediatric patients in order to identify whether or not they have pneumonia. The entire dataset comes from Kermany et al. on [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3), although there is also a version on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) that may be easier to use.

This task is to:

        Build a model that can classify whether a given patient has pneumonia, given a chest x-ray image.

To speed up image

## Loading Models for Visualizing Intermediate Activations of every Deep Learning Model

Deep learning is extremely powerful and is helping to lead the advancement of many AI tasks. That said, deep learning is often criticized for having a lot of black box algorithms in that the components of the model itself are difficult to interpret. In the case of CNNs and image recognition, this is actually not true at all! In this lesson, you will explore how you can visualize the intermediate hidden layers within your CNN to uncover what sorts of features your deep network is uncovering through some of the various filters. With that, you'll gain interesting insights and knowledge as to how your CNN is seeing the world. 

Now that my own CNN has been built so that the visualization of feature maps is able to be seen. It's time to load pretrained models from files and visualize learned features systematically. Here, I'll succinctly visualize all the channels from each layer in a CNN. Saved models such as `'chest_xray_all_data.h5'` and `***.h5` are uploaded to one subdirectory, **models**, of this repository. All of these saved model files include both the model architecture and the trained weights. See the `model.save()` method for further details. These models were built in order to help identify patients with pneumonia. Start simply by loading them and pulling up a summary of the layers from each given pre-trained model. (To load the model use the `keras.models.load_model()` function.) 

## Loading Pretrained Models

In this lesson, you'll start to investigate how to use pretrained networks. Recall that when training neural networks, the model is tuning a huge number of weights: several to dozens at each individual layer. Often the largest limiting factor with these models is the quality and size of the training data you have at hand. As such, adapting a pretrained model from a similar problem context that was trained on a larger dataset can lead to stronger overall models when you have limited training data. For example, in image recognition, the VGG-19 network is commonly used to improve the model performance of CNNs with limited training data. VGG-19 was trained on the ImageNet dataset which contains approximately 1.2 million images. Since the initial bottom layers of a CNN pick up small details with later layers picking up larger and larger features, the initial layers of a well trained network are applicable to other problem domains. Similar pretrained networks exist for other domains such as natural language processing as well. With that, let's take a further look at how transfer learning works in detail.

    Pretrained models are adapted for classifying `PNEUMONIA` or `NORMAL` problem scenario that I've worked on so far!
    
## Tuning Models for the Optimization of Chosen Metrics:

- Early stopping: through varying epochs, the first point at which the local minimal loss appears can be identified.  
- Trainging optimizers: 
- Regularizers: the addition of first- and second-order regularization terms into the cost function to smoothen the variation of both cost and accuracy with time
- Dropout:
- Activation functions:
- Batch size:
- Learning rate:

## Comparison of Model Performances:

## Conclusions:
