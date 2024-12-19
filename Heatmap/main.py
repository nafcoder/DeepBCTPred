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
import matplotlib.pyplot as plt


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


def generate_heatmap(model, input_image, features, target_class, original_image):
    model.eval()

    # Lists to store activations and gradients
    gradients = []
    activations = []

    # Hooks to capture activations and gradients
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Register hooks to the last convolutional layer
    last_conv_layer = model.block.conv_stem  # Modify layer based on architecture
    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_backward_hook(backward_hook)

    try:
        # Forward pass
        input_image.requires_grad_()
        output = model(input_image, features)  # Add batch dimension
        pred_class = output.argmax(dim=1).item()
        print(f"Predicted class: {pred_class}")

        # Backward pass for the target class
        model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Get gradients and activations
        gradients_np = gradients[0].detach().numpy()
        activations_np = activations[0].detach().numpy()

        # Compute the weights for the heatmap
        weights = np.mean(gradients_np, axis=(2, 3))  # Global average pooling
        heatmap = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations_np, axis=1)[0]

        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU to ignore negative values
        heatmap /= np.max(heatmap)  # Scale between 0 and 1

        # Resize the heatmap to the size of the input image
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # Convert heatmap to RGB for overlay
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

        print(heatmap.shape)

       # Plot the heatmap and overlay
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original image
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image", fontsize=20, fontweight='bold')
        axs[0].axis("off")

        # Plot the heatmap
        heatmap_plot = axs[1].imshow(heatmap, cmap="hot")
        axs[1].set_title("Heatmap", fontsize=20, fontweight='bold')
        axs[1].axis("off")
        # Add a colorbar to the heatmap
        cbar = fig.colorbar(heatmap_plot, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Intensity", fontsize=18)

        # Plot the overlay
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay", fontsize=20, fontweight='bold')
        axs[2].axis("off")

        # Set a global title
        # fig.suptitle("Visualization of Image, Heatmap, and Overlay", fontsize=20, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

        # Display the plot
        plt.show()

    finally:
        # Ensure hooks are removed to prevent memory leaks
        forward_handle.remove()
        backward_handle.remove()



def create_heatmap(image_path, target_class):
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


    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)
    features = torch.tensor(features).unsqueeze(0)

    print(image.shape)

    generate_heatmap(model, image, features, target_class, original_image)



num_classes = 4
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load("DeepBCTPred.pth"))

model.eval()

create_heatmap('sample_1.png', 0)
create_heatmap('sample_2.png', 1)
create_heatmap('sample_3.png', 2)
create_heatmap('sample_4.png', 3)
