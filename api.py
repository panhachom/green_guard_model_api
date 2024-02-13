import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models as torch_models
from green_guard import ConvolutionalNetwork

app = Flask(__name__)

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224))   , 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained model
model = ConvolutionalNetwork.load_from_checkpoint("main_model.ckpt")
model.eval()

# Define dataset classes
dataset_path = "Rice Leaf Disease Dataset"
dataset = datasets.ImageFolder(root=dataset_path)



@app.route('/predict', methods=['POST'])

# def predict():
    # Check if the request contains an image file
    # Check if the file is empty

    # Open and preprocess the image
    # apply tranform
    # Add batch dimension

    # Perform inference
    
    # Convert to probabilities and convert to list

    # Create dictionary with class names and corresponding probabilities
  


@app.route('/')
def index():
    return 'Welcome to the Green Guard API!'

if __name__ == '__main__':
    app.run(debug=True)
