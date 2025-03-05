# CIFAR-10 Image Classification with CNN and Transfer Learning

This repository contains two models for classifying CIFAR-10 images:  
1. **Custom CNN model** - A convolutional neural network trained from scratch using tensorflow.  
2. **Pre-trained VGG16 model** - Uses transfer learning from pre-trained VGG16 to improve accuracy.

---

## **Project Overview**
The goal of this project is to classify images from the CIFAR-10 dataset into 10 categories:  
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck.


---

## **Dataset**
- **Dataset Used**: CIFAR-10
- **Training Samples**: 50,000
- **Test Samples**: 10,000
- **Image Size**: 32x32 pixels, RGB

---

## **Model Architectures**
### **Custom CNN (`model.py`)**
- 4 Convolutional Layers with Batch Normalization & Max Pooling
- Batch Normalization
- Data Augmentation (Random Flip, Rotation, Zoom, Contrast)
- Dense Fully Connected Layers with Dropout Regularization

### **Pre-trained VGG16 (`preTrained.py`)**
- Uses **VGG16** as a referecne
- Removed all layers excpet the convolutional and pooling
- Changed the input layer to fit the CIFAR-10 dataset
- Unfreezes the last 4 layers for fine-tuning
- 2 Fully connected layers added for classification

---

##  **Setup & Installation**
### **1. Install Dependencies**
```sh
pip install tensorflow numpy matplotlib scikit-learn
