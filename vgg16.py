#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, callbacks
from torchvision.datasets import ImageFolder
from PIL import Image
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import GridSearchCV  # Add this import
import torchvision.models as models 
import torch.optim as optim  # Import the optim module
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import seaborn as sns
import requests
from io import BytesIO  # Import BytesIO class from io module


# In[2]:


dataset_path = "/kaggle/input/cadt-rice-leaf-disease-image-dataset/Rice Leaf Disease Dataset"
# dataset_path = "/kaggle/input/augmented-dataset/augumented_dataset"

dataset_path


# In[3]:


dataset=datasets.ImageFolder(root=dataset_path)
dataset.classes


# In[4]:


transform = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 360)),  # Random rotation between 0 and 360 degrees (inclusive)
    transforms.RandomHorizontalFlip(p=0.5),  # Optional: Random horizontal flip
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[5]:


# In[7]:


import torch
from torchvision.models import vgg16

class PretrainedModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PretrainedModel, self).__init__()
        self.backbone = vgg16(pretrained=pretrained)  # Load the VGG16 model

        # Extract the number of input features from the last layer of the classifier
        in_features = self.backbone.classifier[6].in_features

        # Replace the last layer with a custom linear layer for the desired number of classes
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Pass the input through the VGG16 backbone
        x = self.backbone(x)
        return x  # Return the output of the backbone (can be modified for specific tasks)

# Create an instance of the PretrainedModel class
model = PretrainedModel(num_classes=1000)  # Assuming 1000 classes for ImageNet

# Print the model architecture
print(model)


# In[ ]:





# In[8]:


class PretrainedDataModule:
    def __init__(self, root_dir, batch_size):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.classes = None

    def setup(self):
        dataset = ImageFolder(root=self.root_dir, transform=self.transform)
        self.classes = dataset.classes
        n_data = len(dataset)
        n_train = int(0.7 * n_data)
        n_valid = int(0.15 * n_data)
        n_test = n_data - n_train - n_valid

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return len(dataset.classes)


# In[9]:


if __name__ == "__main__":
     # Initialize the PretrainedDataModule
    data_module = PretrainedDataModule(root_dir=dataset_path, batch_size=32)
    # Setup the data module and get the number of classes
    num_classes = data_module.setup()
    # Initialize the PretrainedModel
    model = PretrainedModel(num_classes=num_classes, pretrained=True)
    # Print the model architecture
    print(model)


# In[10]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[11]:


# Initialize TensorBoard for visualization
writer = SummaryWriter()


# In[12]:


# Training loop for fine-tuning
num_epochs_finetune = 10
for epoch in range(num_epochs_finetune):
    model.train()
    running_loss = 0.0
    for images, labels in data_module.train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_module.train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs_finetune}], Fine-tune Train Loss: {epoch_loss:.4f}")


# In[13]:


# Evaluation on validation set after fine-tuning
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_module.valid_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy after Fine-tuning: {accuracy:.4f}")


# In[14]:


# Evaluate the model on the test set
model.eval()
y_true = []
y_pred = []
all_predictions = []
all_labels = []
with torch.no_grad():
    for images, labels in data_module.test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())


# In[15]:


# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)


# In[16]:


# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=data_module.classes))


# In[17]:


# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data_module.classes, yticklabels=data_module.classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()




# In[19]:


torch.save(model.state_dict(), 'VGG16.pth')




# Download the image from the internet

import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Initialize the model
model = PretrainedModel(num_classes=num_classes, pretrained=True)

# Load the model's parameters
model.load_state_dict(torch.load('VGG16.pth'))
model.eval()

# Define the threshold
best_threshold = 0.8

# Download the image from the internet
image_url = "https://source.roboflow.com/l8N0UZkZfcaMxao7DB0aaiEhww93/6f98ubrZ331n7Li9dSmI/thumb.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Preprocess the image
preprocessed_image = transform(image)

# Make predictions
with torch.no_grad():
  output = model(preprocessed_image.unsqueeze(0)) # Unsqueeze to add batch dimension
  probabilities = torch.softmax(output, dim=1)
  predicted_prob, predicted_class = torch.max(probabilities, 1)

# Apply thresholding
if predicted_prob.item() > best_threshold:
  # Prediction above threshold
  predicted_label = data_module.classes[predicted_class.item()]
  print(f"Predicted Label: {predicted_label}, Probability: {predicted_prob.item()}")
else:
  print("Prediction below threshold. Considered invalid.")


