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

## 2. Transformation Pipeline

To enhance the model's generalization, a data transformation pipeline is applied to each image. This pipeline includes the following steps:

1. **Random Crop**: Images are cropped to 196x196 pixels, providing a consistent input size.
2. **Random Rotation**: Images are randomly rotated by up to 15 degrees to add variability.
3. **Random Horizontal Flip**: Images are flipped horizontally with a 50% probability to introduce further variability.
4. **Color Jitter**: Adjusts the brightness, contrast, saturation, and hue slightly to help the model handle variations in lighting and colors.
5. **Normalization**: The image is normalized to align with ImageNet's mean and standard deviation values, matching the ResNet50’s pre-trained parameters.


## 3. Training Process

### Hyperparameters

The model is trained using the following hyperparameters:

- **Loss Function**: Cross-Entropy Loss, ideal for multi-class classification tasks.
- **Optimizer**: Adam optimizer with a learning rate of `5e-5` and a weight decay of `0.005`, aimed at controlling overfitting.
- **Epochs**: The model was trained over 3 epochs to minimize classification errors progressively.
