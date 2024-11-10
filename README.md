# Custom CNN Model Summary Report

This report provides an overview of the `CustomCNN` model, the transformation pipeline, training process, and results. The `CustomCNN` model is built on a pre-trained ResNet50, fine-tuned for binary classification tasks.

## 1. Model Architecture

The `CustomCNN` model leverages a pre-trained ResNet50 model as its base and enables training on all layers to fine-tune it for a specific classification task. The model removes the original final layer from ResNet50 and replaces it with custom fully connected layers:

- **Base Model**: ResNet50 pre-trained on ImageNet, with all layers set to be trainable.
- **Dropout Layer**: A dropout layer with a probability of 0.2 is added to prevent overfitting.
- **Fully Connected Layers**:
  - First layer reduces the feature size to 128.
  - ReLU activation function follows to introduce non-linearity.
  - Second layer reduces the output to 2, suitable for binary classification.


