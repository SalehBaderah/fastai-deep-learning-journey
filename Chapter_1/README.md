# Chapter 1: Deep Learning Fundamentals

## 1. What is Deep Learning?
> **Definition:** A computer technique used to extract and transform data for use cases such as human speech recognition, animal image classification, etc.

It works by using multiple layers of **Neural Networks**. Each of these layers takes input from the previous layer and refines it by minimizing errors and improving accuracy.

### Common Tasks of Deep Learning
* **NLP (Natural Language Processing):** Answering questions, speech recognition, summarizing documents, classifying text, finding names/dates in documents, and searching for concepts.
* **Computer Vision:** Satellite and drone imagery interpretation (e.g., disaster resilience), face recognition, image captioning, reading traffic signs, and locating pedestrians/vehicles for autonomous driving.
* **Medicine:** Finding anomalies in radiology images (CT, MRI, X-ray), counting features in pathology slides, measuring features in ultrasounds, and diagnosing diabetic retinopathy.
* **Image Generation:** Colorizing black-and-white images, increasing image resolution, removing noise, or converting images into the style of famous artists.
* **Other Applications:** Financial and logistical forecasting, text-to-speech, and much more.

### The Goal
To build a machine capable of learning from data to perform a task (training) or controlling a system.

## 2. Tools
* **Jupyter Notebooks:** The interactive coding environment.
* **fastai:** The high-level library for rapid implementation.
* **PyTorch:** The underlying deep learning framework.
* **[Book Website](https://course.fast.ai/Lessons/lesson1.html):** Source for code and resources.

---

## 3. Core Concepts

### What is Machine Learning?
Allowing a computer or algorithm to learn from data rather than being explicitly programmed with rules.

### What is a Neural Network?
A mathematical function that mimics the human brain. It uses **weights** that are optimized automatically using **Stochastic Gradient Descent (SGD)** (covered in Chapter 4) to find the optimal accuracy.

---

## 4. Terminology & Definitions

### Data Splits
* **Training Set (~80%):** Used for **training** the model.
* **Validation Set (~20%):** Used to measure the **accuracy** of the model (i.e., "How well is the model doing on data it hasn't seen?").
    * **Seed:** A random number generator setting (e.g., `seed=42`). We set this to ensure we get the *same* validation set every time we run the code, ensuring reproducibility.

### Model Behavior
* **Overfitting:** When the model "memorizes" the specific images in the training set rather than learning the **underlying patterns**. This leads to poor performance on the validation set.
* **Metric:** A function used to measure the quality of the model's predictions. It is calculated and displayed at the end of each **epoch** (e.g., Accuracy, Error Rate).
* **Epoch:** One complete pass through all the images in the dataset.

<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-11%20151250.png" width="400" alt="Overfitting Graph">
</div>

### Transfer Learning
* **Transfer Learning:** When we use a **Pretrained Model** for a task different from what it was originally trained for.
* **Fine-Tuning:** The process of training a pretrained model for additional epochs on your specific dataset to adapt the weights to your new task.

#### Code Example:
```python
# 'dls' is your DataLoaders
# 'resnet34' is the pretrained model architecture
# 'error_rate' is the metric we want to see
learn = cnn_learner(dls, resnet34, metrics=error_rate)
# Fine-tune the model for 1 epoch 
learn.fine_tune(1)
    
```
---
# Recap
<div align="center">
  <img src="https://github.com/SalehBaderah/fastai-deep-learning-journey/blob/main/Images/Screenshot%202026-01-11%20165328.png" width="500" alt="Recap">
</div>


# Validation Sets & Test Sets
The problem with validation set is the `Data Leakage` hapens when we train the model more than once and update the hyperparameter until we find the best validation score.
- Round 1: You try a "Learning Rate" of 0.1. Validation accuracy is 80%.

- Round 2: You try a "Learning Rate" of 0.01. Validation accuracy is 82%.

- Round 3: You try a "Learning Rate" of 0.001. Validation accuracy is 85%.

  By choosing the settings that performed best on validation set, we indirectly "LEAKED" information from the validation set. The model now is Biased toward it .

The solution : The `Test set`
this set is totaly hidden, It cannot be used to improve the model, Used only to evaluate the model at the end.
> Both validation set and Test set should have enough data to ensure that we get good estimate of the accuracy
> In case we don't have enough data we may need just validation set
