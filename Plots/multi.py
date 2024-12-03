import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
plt.rcParams.update({'font.size': 12.5})
# Load true labels and probabilities
def load_data(y_true_path, y_proba_path):
    y_true = np.load(y_true_path)
    y_proba = np.load(y_proba_path)
    return y_true, y_proba

# Convert probabilities to predicted labels
def probabilities_to_predictions(y_proba):
    return np.argmax(y_proba, axis=1)

# Calculate additional metrics for each class
def calculate_metrics(y_true, y_proba, n_classes):
    y_pred = probabilities_to_predictions(y_proba)
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))
    y_proba_binarized = y_proba  # `y_proba` is already multi-class probabilities
    
    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
        "specificity": [],
        "MCC": [],
        "AUPR": [],
        "AUROC": []
    }
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    overall_accuracy = np.trace(confusion) / np.sum(confusion)

    for i in range(n_classes):
        cls = str(i)
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = np.sum(confusion) - (TP + FP + FN)

        # Metrics for each class
        precision = report[cls]["precision"]
        recall = report[cls]["recall"]
        f1_score = report[cls]["f1-score"]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        MCC = matthews_corrcoef(y_true == i, y_pred == i)
        AUPR = average_precision_score(y_true_binarized[:, i], y_proba_binarized[:, i])
        AUROC = roc_auc_score(y_true_binarized[:, i], y_proba_binarized[:, i])

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1_score)
        metrics["specificity"].append(specificity)
        metrics["MCC"].append(MCC)
        metrics["AUPR"].append(AUPR)
        metrics["AUROC"].append(AUROC)
    
    metrics["accuracy"] = [overall_accuracy] * n_classes  # Same accuracy for all classes
    return metrics

# Plot metrics
def plot_metrics(metrics, class_names):
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.15  # Width of bars
    colors = ["blue", "green", "orange", "red"]  # Different colors for classes

    # Initialize figure
    fig, axes = plt.subplots(2, 4, figsize=(8, 6), sharey=False)
    axes = axes.flatten()

    # Define metric names for plotting
    metric_names = ["recall", "specificity", "accuracy", "precision", "f1_score", "MCC", "AUROC", "AUPR"]
    real_metric_names = ["REC", "SPEC", "ACC", "PREC", "F1", "MCC", "AUC", "AUPR"]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        for i in range(n_classes):
            ax.bar(
                x[i] + (i - n_classes // 2) * width,  # Offset for each class
                metrics[metric][i],
                width,
                label=class_names[i] if idx == 0 else "",  # Add legend only for the first subplot
                color=colors[i]
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_title(real_metric_names[idx])
        ax.set_xlabel("Classes")
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(0.8, 1)  # Show range from 0.8 to 1

    fig.legend(loc="upper center", ncol=n_classes, fancybox=True, shadow=True, title="Classes")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the legend
    plt.show()

# Load data for DeepBCTPred
class_names = ["HGC", "LGC", "NST", "NTL"]  # Class labels
n_classes = len(class_names)
y_true_path = "Results/DeepBCTPred_y_true.npy"
y_proba_path = "Results/DeepBCTPred_y_proba.npy"

y_true, y_proba = load_data(y_true_path, y_proba_path)

# Calculate metrics for DeepBCTPred
metrics = calculate_metrics(y_true, y_proba, n_classes)

# Plot metrics for DeepBCTPred
plot_metrics(metrics, class_names)
