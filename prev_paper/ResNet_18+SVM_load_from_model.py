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
import torchvision.models as models
import numpy as np
import timm
import torch
import torch.nn.functional as F
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassAUPRC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import random
from sklearn.svm import SVC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predicted, probabilities, labels, num_classes):
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
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"Specificity per class: {specificity_per_class}")
    print(f"Macro-average Specificity: {macro_specificity:.4f}")
    print(f"MCC per class: {mcc_per_class}")
    print(f"Macro-average MCC: {macro_mcc:.4f}")

    f1_score_metric_micro = MulticlassF1Score(num_classes=num_classes, average="micro")
    f1_score_metric_micro.update(predicted, labels)
    f1_score_micro = f1_score_metric_micro.compute()
    print(f"F1 Score (micro): {f1_score_micro:.4f}")

    f1_score_metric_weighted = MulticlassF1Score(num_classes=num_classes, average="weighted")
    f1_score_metric_weighted.update(predicted, labels)
    f1_score_weighted = f1_score_metric_weighted.compute()
    print(f"F1 Score (weighted): {f1_score_weighted:.4f}")


    # Return metrics as a dictionary
    return {
        "accuracy": "{:.4f}".format(accuracy.item()),
        "precision": "{:.4f}".format(precision.item()),
        "recall": "{:.4f}".format(recall.item()),
        "f1_score": "{:.4f}".format(f1_score.item()),
        "auc": "{:.4f}".format(auc.item()),
        "aupr": "{:.4f}".format(aupr.item()),
        "macro_specificity": "{:.4f}".format(macro_specificity.item()),
        "macro_mcc": "{:.4f}".format(macro_mcc.item())
    }


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

set_seed(42)

dataset_path = "baldder_tissue_classification"

test_df = pd.read_csv("test_df.csv")

test_images = []
test_labels = []
for idx, row in test_df.iterrows():
    image_path = os.path.join(dataset_path, row[1], row[0])
    image = Image.open(image_path)
    image = transform(image)
    test_images.append(image)
    test_labels.append(row['encoded_label'])

print(len(test_images))


# Instantiate model, loss function, and optimizer
num_classes = 4
model = models.resnet18(pretrained=True)

model.eval()
# inference with batch size 8
test_features = []

for i in range(0, len(test_images), 8):
    batch_images = test_images[i:i+8]
    batch_images = torch.stack(batch_images)
    with torch.no_grad():
        features = model(batch_images)
    test_features.extend(features.detach().cpu().numpy())


test_features = np.array(test_features)
test_labels = np.array(test_labels)

print(test_features.shape)
print(test_labels.shape)

svm = SVC(kernel="linear", decision_function_shape="ovo", probability=True, random_state=42)
# load from pickle
import pickle
with open('Models/ResNet_18+SVM.pkl', 'rb') as f:
    svm = pickle.load(f)
predicted = svm.predict(test_features)
probabilities = svm.predict_proba(test_features)

# Calculate metrics

metrics = calculate_metrics(torch.tensor(predicted), torch.tensor(probabilities), torch.tensor(test_labels), num_classes)
print(metrics)

