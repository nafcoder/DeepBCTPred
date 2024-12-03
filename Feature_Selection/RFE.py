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
from sklearn.feature_selection import RFE
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


def getModel(model_name):
    models = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, multi_class="ovr"),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
        "LDA (Linear Discriminant Analysis)": LinearDiscriminantAnalysis(),
        "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(objective="multi:softmax", num_class=3, use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "SVM": SVC(kernel="rbf", C=1, decision_function_shape="ovr", probability=True, random_state=42),
    }
    if model_name in models:
        return models[model_name]



set_seed(42)

# Load the annotation CSV
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")

# merge train and val
train_df = pd.concat([train_df, val_df])

X = np.load("features.npy")
y = train_df["encoded_label"].values
print(X.shape)
print(y.shape)


lgbm = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbose=-1)
rfe = RFE(estimator=lgbm, n_features_to_select=512, step=200, verbose=1)

# Fit RFE to the data
rfe.fit(X, y)

# Get selected features
selected_features = rfe.support_

# Save the selected features
np.save("selected_features.npy", selected_features)
