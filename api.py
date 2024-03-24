
import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models as torch_models
from resnet_18 import PretrainedModel

app = Flask(__name__)
# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your trained model
model = PretrainedModel( num_classes=7)  

model.load_state_dict(torch.load('resnet_18_model.pth'))
model.eval()  # Set the model to evaluation mode


# Define dataset classes
dataset_path = "Rice Leaf Disease Dataset"
dataset = datasets.ImageFolder(root=dataset_path)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found in request'}), 400

        image_file = request.files['image']

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'error': 'Image file is empty'}), 400

        try:
            img = Image.open(image_file.stream)
            img_tensor = transform(img).unsqueeze(0)  # Apply transforms and add batch dimension

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)
                probabilities = torch.nn.functional.softmax(predictions, dim=1)[0].tolist()  # Convert to probs

            # Retrieve class names from dataset
            class_names = dataset.classes

            # Create dictionary with class names and corresponding probabilities
            prediction_dict = {class_name: round(prob * 100, 2) for class_name, prob in zip(class_names, probabilities)}

            # return jsonify({'predictions': prediction_dict}), 200
            return jsonify(prediction_dict), 200

        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    else:
        return jsonify({'error': 'Invalid request method'}), 405
     
@app.route('/')
def index():
    return 'Welcome to the Green Guard API!'

if __name__ == 'main':
    app.run(debug=True)