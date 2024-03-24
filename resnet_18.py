#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os
os.environ['TORCH_MODEL_ZOO'] = '1'
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
from sklearn.model_selection import GridSearchCV  
import torchvision.models as models 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter  
import seaborn as sns
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# In[6]:


dataset_path = "Rice Leaf Disease Dataset"
# dataset_path = "/kaggle/input/augmented-dataset/augumented_dataset"

dataset_path


# In[7]:


dataset=datasets.ImageFolder(root=dataset_path)
dataset.classes


# In[8]:


# Define transform
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])





# In[9]:


class DataModule(pl.LightningDataModule):
    
    def __init__(self, transform=transform, batch_size=32):
        super().__init__()
        self.root_dir = dataset_path
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        n_data = len(dataset)
        n_train = int(0.7 * n_data)
        n_valid = int(0.15 * n_data)
        n_test = n_data - n_train - n_valid

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])

        self.trainset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=3)
        self.validset = DataLoader(valid_dataset, batch_size=self.batch_size ,num_workers=3)
        self.testset = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=3)

    def train_dataloader(self):
        return self.trainset
    
    def val_dataloader(self):
        return self.validset
    
    def test_dataloader(self):
        return self.testset
        return DataLoader(self.testset, batch_size=self.batch_size)


# In[10]:

# In[11]:


class PretrainedModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PretrainedModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout with p=0.5
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# In[12]:


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


# In[13]:


if __name__ == "__main__":
     # Initialize the PretrainedDataModule
    data_module = PretrainedDataModule(root_dir=dataset_path, batch_size=32)
    # Setup the data module and get the number of classes
    num_classes = data_module.setup()
    # Initialize the PretrainedModel
    model = PretrainedModel(num_classes=num_classes, pretrained=True)
    # Print the model architecture
    print(model)


    # In[14]:


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    # In[15]:


    # Initialize TensorBoard for visualization
    writer = SummaryWriter()


    # In[16]:


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


    # In[17]:


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


    # In[35]:


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


    # In[36]:


    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)


    # In[37]:


    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=data_module.classes))


    # In[38]:


    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data_module.classes, yticklabels=data_module.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

