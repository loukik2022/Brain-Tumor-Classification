# Brain Tumor Classification using Deep Learning

This project implements a Convolutional Neural Network (CNN) to classify different types of brain tumors from MRI images. The model can distinguish between four categories: glioma, meningioma, no tumor, and pituitary tumors.

The project uses PyTorch to build and train a custom CNN model for classifying brain tumor MRI images. 
The model achieves 96.57% accuracy on the test set, making it a reliable tool for preliminary tumor classification.

## Dataset

The dataset consists of MRI scans divided into four categories: 
- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The training set contains approx. 7,000 images across all categories.
- source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Model Architecture

The CNN model (`TumorClassifier`) consists of:

### Feature Extraction Layers:
```python
- Conv2d(3, 16, kernel_size=3, padding=1)
- ReLU
- MaxPool2d(kernel_size=2, stride=2)
- Conv2d(16, 32, kernel_size=3, padding=1)
- ReLU
- MaxPool2d(kernel_size=2, stride=2)
```

### Classification Layers:
```python
- Linear(32 * 56 * 56, 128)
- ReLU
- Linear(128, 4)
```

## Training
- Batch Size: 16
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Cross Entropy Loss
- Early Stopping: Patience of 5 epochs


## Results:
- Accuracy: 96.57%
- Training completed in 13 epochs with early stopping

![Loss-Accuracy History]('results.png')