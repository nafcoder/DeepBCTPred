import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.measure import shannon_entropy
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassAUPRC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import timm
import torch
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
import pickle
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def lbp_features(image, radius=1, n_points=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return lbp.flatten()


def haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray)  # Convert to unsigned integer type if necessary
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features.append(graycoprops(glcm, prop)[0, 0])
    return features


def color_histogram(image):
    chans = cv2.split(image)
    hist_features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist_features.extend(hist.flatten())
    return np.array(hist_features)


def dominant_colors(image, k=3):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k).fit(data)
    colors = kmeans.cluster_centers_
    return colors


def edge_histogram(image):
    edges = cv2.Canny(image, 100, 200)
    hist, _ = np.histogram(edges.ravel(), bins=256, range=(0, 256))
    return hist


def hog_features(image):
    # Apply HOG with channel_axis to handle RGB images
    hog_features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1), visualize=True, channel_axis=2)
    return hog_features


def sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros((128,))  # Return zero vector if no keypoints
    
    # Compute the mean of the descriptors
    return np.mean(descriptors, axis=0)


def orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros((32,))  # ORB descriptors are 32-dimensional

    # Compute the mean of the descriptors
    return np.mean(descriptors, axis=0)



def pixel_intensity_stats(image):
    mean, stddev = cv2.meanStdDev(image)
    return mean.flatten(), stddev.flatten()


def image_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)


def dct_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray) / 255.0)
    return dct.flatten()


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_1(image)  # This is now a tensor
    image = image.numpy()  # Convert tensor to numpy array
    image = (image * np.array([0.229, 0.224, 0.225])[:, None, None]) + np.array([0.485, 0.456, 0.406])[:, None, None]
    image = (image * 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8
    image = image.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    
    ch = color_histogram(image).reshape(-1)
    dc = dominant_colors(image).reshape(-1)
    hf = torch.tensor(haralick_features(image)).reshape(-1)
    lbp = lbp_features(image).reshape(-1)
    eh = edge_histogram(image).reshape(-1)
    hog = hog_features(image).reshape(-1)
    sf = sift_features(image).reshape(-1)
    of = orb_features(image).reshape(-1)

    mean, std_dev = pixel_intensity_stats(image)
    ms = torch.tensor(np.concatenate((mean, std_dev)), dtype=torch.float32)

    en = torch.tensor(image_entropy(image)).reshape(-1)
    dct = dct_features(image).reshape(-1)

    features = np.concatenate((ch, dc, hf, lbp, eh, hog, sf, of, ms, en, dct))
    return features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


transform_1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


set_seed(42)

image_path = 'sample_4.png'

features = np.array(extract_features(image_path))

# save pt with pickle
with open('power_transformer.pkl', 'rb') as f:
    pt = pickle.load(f)

# add batch dimension
features = features.reshape(1, -1)

# save transformed data
features = pt.transform(features)
features = features.reshape(-1)

selected_features = np.load("selected_features.npy")
features = features[selected_features]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.block = timm.create_model('tf_mobilenetv3_large_minimal_100.in1k', pretrained=True)
        # self.block.head = nn.Linear(self.block.num_features, 1000)

        self.fc1 = nn.Linear(1000, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc_ann1 = nn.Linear(512, 256)
        self.bn_ann1 = nn.BatchNorm1d(256)
        self.relu_ann1 = nn.ReLU()
        self.dropout_ann1 = nn.Dropout(0.2)
        self.fc_ann2 = nn.Linear(256, 128)
        self.bn_ann2 = nn.BatchNorm1d(128)
        self.relu_ann2 = nn.ReLU()
        self.dropout_ann2 = nn.Dropout(0.2)

        self.fc_comb = nn.Linear(384, 256)
        self.bn_comb = nn.BatchNorm1d(256)
        self.relu_comb = nn.ReLU()
        self.dropout_comb = nn.Dropout(0.2)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        

    def forward(self, x, features):
        features = features.float()
        x = self.block(x)

        x = x.view(-1, 1000)  # Flatten
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))

        features = self.dropout_ann1(self.relu_ann1(self.bn_ann1(self.fc_ann1(features))))
        features = self.dropout_ann2(self.relu_ann2(self.bn_ann2(self.fc_ann2(features))))

        x = torch.cat((x, features), dim=1)
        x = self.dropout_comb(self.relu_comb(self.bn_comb(self.fc_comb(x))))

        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x



num_classes = 4
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load("DeepBCTPred.pth"))

model.eval()

image = Image.open(image_path)
image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)
features = torch.tensor(features).unsqueeze(0)

outputs = model(image, features)

# Apply softmax to get probabilities
probabilities = F.softmax(outputs, dim=1)

proba, predicted = torch.max(probabilities, 1)

label_mapper = {0: 'HGC', 1: 'LGC', 2: 'NST', 3: 'NTL'}


print("Predicted class:", label_mapper[predicted.item()])
print("Probability:", proba.item())
