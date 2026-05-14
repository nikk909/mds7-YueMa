# Titanic Deep Learning Pipeline Report

## 1. Model Architectures
This pipeline evaluates two distinct Neural Network topologies for binary classification.

### Model A: Shallow Network (3 Layers)
- **Structure**: 1 Input/Hidden Layer, 1 Output Layer.
- **Neuron Density**: 16 neurons in the hidden layer.
- **Activation Functions**: ReLU for hidden, Sigmoid for output.
- **Final Test Accuracy**: 0.7933

### Model B: Deep Network (5 Layers)
- **Structure**: 1 Input Layer, 3 Hidden Layers, 1 Output Layer.
- **Neuron Density**: 32, 16, 8 neurons respectively in hidden layers.
- **Activation Functions**: ReLU for all hidden layers, Sigmoid for output.
- **Final Test Accuracy**: 0.7989

## 2. Pipeline Execution Details
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Framework**: Keras Sequential API
- **Deployment**: Automated dual-cloud upload to AWS S3 and GitHub.
