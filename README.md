# COVID-19-Xray-Dataset--CNN-ResNet50-MobileNetV2
## Introduction
- Dataset Overview: Briefly describe the COVID-19 X-ray dataset. Mention the input data (X-ray images), target labels (COVID-19, normal, pneumonia), and any preprocessing steps.
- Objective: Explain the goal of using the dataset. For example, detecting COVID-19 from X-ray images using deep learning models
## Tools and Libraries
This project uses a combination of deep learning frameworks and utilities for image processing, model building, and evaluation. Below is an overview of the key tools and libraries used:

### 1.1 TensorFlow and Keras
- TensorFlow: A powerful open-source platform for building machine learning models. In this project, TensorFlow is used to implement and train deep learning models such as MobileNetV2 and ResNet50.
- Keras (within TensorFlow): Keras provides a high-level API for building and training neural networks. Keras layers and models make it easier to build custom CNN architectures.
### 1.2 Keras Models and Layers
- Sequential Model: This is used for stacking layers sequentially to create custom Convolutional Neural Networks (CNNs).
- Layers: Layers like Conv2D, MaxPool2D, Dropout, Flatten, and Dense are used to build the structure of the CNN.
- Pre-trained Models (ResNet50 and MobileNetV2): Used for transfer learning. These models come with pre-trained weights and are adapted to the X-ray dataset.
### 1.3 Image Processing and Data Augmentation
- ImageDataGenerator: This Keras utility is used for real-time data augmentation, which helps prevent overfitting by applying transformations like rotations, flips, and rescaling to the input images.
### 1.4 Optimization and Loss Functions
- Adam Optimizer: This optimization algorithm is used for updating the weights of the neural network during training.
### 1.5 Evaluation Metrics
- Confusion Matrix and Classification Report: Provided by sklearn.metrics, these are used for evaluating model performance by calculating precision, recall, and F1-score for each class.
python
### 1.6 Warnings Handling
- Warnings Module: This is used to suppress unnecessary warnings that may clutter the output during training and evaluation.
## Deep Learning Models:
