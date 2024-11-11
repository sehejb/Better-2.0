from flask import Flask, request, jsonify
import torch, pickle, torchvision
from DeepfakeDetectionModel import CustomCNN
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import transforms, models
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.nn.functional as F
import PIL

app = Flask(__name__)

# Load the entire model
model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    # Preprocess input data
    inputs = torch.tensor(input_data['inputs'])
    # Make predictions
    outputs = model(inputs)
    # Postprocess output data
    predictions = outputs.detach().numpy().tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
