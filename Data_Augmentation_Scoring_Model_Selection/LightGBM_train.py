import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import warnings
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassAUPRC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predicted, probabilities, labels, num_classes):
    predicted = predicted.to(torch.int64)
    labels = labels.to(torch.int64)
    # Initialize Torcheval metrics
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro")
    recall_metric = MulticlassRecall(num_classes=num_classes, average="macro")
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="macro")
    auc_metric = MulticlassAUROC(num_classes=num_classes, average="macro")
    aupr_metric = MulticlassAUPRC(num_classes=num_classes, average="macro")

    # Update metrics with predictions
    accuracy_metric.update(predicted, labels)
    precision_metric.update(predicted, labels)
    recall_metric.update(predicted, labels)
    f1_score_metric.update(predicted, labels)
    auc_metric.update(probabilities, labels)
    aupr_metric.update(probabilities, labels)

    # Compute metrics
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1_score = f1_score_metric.compute()
    auc = auc_metric.compute()
    aupr = aupr_metric.compute()

    # Specificity calculation
    conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))
    specificity_per_class = []

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_per_class.append(specificity)

    macro_specificity = np.mean(specificity_per_class)

    # MCC calculation for multi-class classification
    mcc_per_class = []
    for i in range(num_classes):
        # Treat each class as the positive class and others as negative
        binary_pred = (predicted == i).int()
        binary_labels = (labels == i).int()
        mcc = matthews_corrcoef(binary_labels.cpu().numpy(), binary_pred.cpu().numpy())
        mcc_per_class.append(mcc)

    macro_mcc = np.mean(mcc_per_class)

    # Print all metrics
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1_score:.4f}")
    # print(f"AUC: {auc:.4f}")
    # print(f"AUPR: {aupr:.4f}")
    # print(f"Specificity per class: {specificity_per_class}")
    # print(f"Macro-average Specificity: {macro_specificity:.4f}")
    # print(f"MCC per class: {mcc_per_class}")
    # print(f"Macro-average MCC: {macro_mcc:.4f}")

    f1_score_metric_micro = MulticlassF1Score(num_classes=num_classes, average="micro")
    f1_score_metric_micro.update(predicted, labels)
    f1_score_micro = f1_score_metric_micro.compute()
    # print(f"F1 Score (micro): {f1_score_micro:.4f}")

    f1_score_metric_weighted = MulticlassF1Score(num_classes=num_classes, average="weighted")
    f1_score_metric_weighted.update(predicted, labels)
    f1_score_weighted = f1_score_metric_weighted.compute()
    # print(f"F1 Score (weighted): {f1_score_weighted:.4f}")


    # Return metrics as a dictionary
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
        "auc": auc.item(),
        "aupr": aupr.item(),
        "specificity": macro_specificity.item(),
        "mcc": macro_mcc.item()
    }


class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 1], self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)
        
        return image, label



# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

set_seed(42)

# Load the annotation CSV
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")

# merge train and val
train_df = pd.concat([train_df, val_df])

# 10-fold cross-validation for model selection
num_classes = len(train_df["encoded_label"].unique())

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

dataset_path = "baldder_tissue_classification"
train_dataset = ImageDataset(train_df, dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=train_df.shape[0], shuffle=False)

X = []
y = []

# Extract images as NumPy arrays
for images_np, labels in train_loader:
    # Convert each batch of images to NumPy
    images_np = images_np.numpy()  # Convert tensor to NumPy

    # save the images and labels to X and y
    X = images_np.reshape(images_np.shape[0], -1)
    y = labels.numpy()

print(X.shape)
print(y.shape)

model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbose=-1)
model.fit(X, y)

filename = 'LightGBM.pkl'
pickle.dump(model, open(filename, 'wb'))