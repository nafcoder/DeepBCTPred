import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def display_images(images, labels=None, num_images=4):
    # Limit the number of images displayed
    images_to_display = images[:num_images]
    labels_to_display = labels[:num_images] if labels else [None] * num_images
    
    plt.figure(figsize=(15, 5))
    
    for idx, (img, label) in enumerate(zip(images_to_display, labels_to_display)):
        # Undo normalization if required
        img = (img * np.array([0.229, 0.224, 0.225])[:, None, None]) + np.array([0.485, 0.456, 0.406])[:, None, None]
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8
        
        # Convert image to (H, W, C) format if needed
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        elif isinstance(img, np.ndarray) and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
        
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        
        # Add label if provided
        if label is not None:
            plt.title(label, fontsize=28)
    
    plt.tight_layout()
    plt.show()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


set_seed(1)

# Load the dataset
new_images = np.load('top_images.npy')
new_labels = np.load('top_labels.npy')

# randomly select 4 images with different labels
selected_images = []
selected_labels = []
choices = [0, 1, 2, 3]
id = 0
while len(selected_images) < 4:
    idx = random.randint(0, len(new_images) - 1)
    if new_labels[idx] == choices[id]:
        selected_images.append(new_images[idx])
        selected_labels.append(new_labels[idx])
        id += 1

print(selected_labels)
images = []

for i in range(4):
    image = selected_images[i]
    image = (image * np.array([0.229, 0.224, 0.225])[:, None, None]) + np.array([0.485, 0.456, 0.406])[:, None, None]
    image = (image * 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8
    image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    image = Image.fromarray(image)
    image = transform(image)  # This is now a tensor
    image = image.numpy()  # Convert tensor to numpy array

    images.append(image)

display_images(images,  ['HGC', 'LGC', 'NST', 'NTL'])
