# CIFAR-10 Image Classification

This project builds a deep learning model using a fully connected neural network to classify images from the CIFAR-10 dataset. CIFAR-10 contains 60,000 color images across 10 classes, such as airplanes, cars, birds, cats, ships, and trucks. The goal of this project is to understand the full workflow of image classification — from preprocessing to model training, visualization, and evaluation.

## 🚀 Features

- Loads and preprocesses the CIFAR-10 dataset
- Normalizes image pixels to a range of 0-1
- One-hot encodes labels for classification
- Builds a Dense Neural Network with:
  - Multiple hidden layers
  - Batch Normalization
  - Dropout layers to avoid overfitting
- Trains the model using Adam optimizer and categorical crossentropy loss
- Plots:
  - Training vs Validation Accuracy
  - Training vs Validation Loss
- Evaluates test accuracy
- Displays sample predictions with images

## 📦 Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow / Keras
- CIFAR-10 Dataset

## 📁 Project Structure
🧠 Model Summary

The neural network consists of:
- Flatten input layer (32×32×3 → 3072)
- Several Dense layers with ReLU activation
- Batch Normalization layers
- Dropout for regularization
- Output layer with Softmax activation (10 classes)

## 📊 Visualizations

The notebook includes:
- Training vs Validation Accuracy curve
- Training vs Validation Loss curve

These visualizations help in understanding the learning performance and detecting overfitting.

## ✅ Results

- **Final Test Accuracy:** 49%
- The model also displays predictions on sample test images.
