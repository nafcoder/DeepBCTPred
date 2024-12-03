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


test_dataset = ImageDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_images), shuffle=False)

# Instantiate model, loss function, and optimizer
num_classes = 4
model = models.alexnet(pretrained=False)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

model.load_state_dict(torch.load("Models/AlexNet_test.pth"))

model.eval()

y_pred = []
y_true = []
y_proba = []
for images, labels in test_loader:
    outputs = model(images)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)
    
    # Get the predicted class and corresponding probability
    proba, predicted = torch.max(probabilities, 1)

    # Store results
    y_pred.extend(predicted.tolist())
    y_true.extend(labels.tolist())
    y_proba.extend(probabilities.tolist())

# Convert lists to tensors
y_pred = torch.tensor(y_pred)
y_true = torch.tensor(y_true)
y_proba = torch.tensor(y_proba)

metrics = calculate_metrics(y_pred, y_proba, y_true, num_classes)
print(metrics)
