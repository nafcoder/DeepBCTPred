import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
plt.rcParams.update({'font.size': 12.5})

def get_precision_recall(y_true_path, y_scores_path, n_classes=4):
    y_true = np.load(y_true_path)
    y_scores = np.load(y_scores_path)

    # Binarize the true labels for multiclass PR curve computation
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Calculate Precision-Recall and Average Precision for each class
    precision = dict()
    recall = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_scores[:, i])
    
    return precision, recall


def get_interpolated_scores(recall, precision, n_classes=4):
    # Compute macro-averaged Precision-Recall curve
    all_precisions = []

    for i in range(n_classes):
        # Interpolate precision for a consistent recall grid
        interp_recall = np.linspace(0, 1, 100)  # Standard recall grid
        interp_precision = np.interp(interp_recall, recall[i][::-1], precision[i][::-1])  # Reverse to make monotonic
        all_precisions.append(interp_precision)

    # Macro-average of precisions across classes
    macro_precision = np.mean(all_precisions, axis=0)

    return interp_recall, macro_precision

# Simulate some data
n_classes = 4
global_precision = []
global_recall = []

precision, recall = get_precision_recall('Results/ResNet_18+SVM_y_true.npy', 'Results/ResNet_18+SVM_y_proba.npy')
global_precision.append(precision)
global_recall.append(recall)

precision, recall = get_precision_recall('Results/ViT_y_true.npy', 'Results/ViT_y_proba.npy')
global_precision.append(precision)
global_recall.append(recall)

precision, recall = get_precision_recall('Results/AlexNet_test_y_true.npy', 'Results/AlexNet_test_y_proba.npy')
global_precision.append(precision)
global_recall.append(recall)

precision, recall = get_precision_recall('Results/17DCNN_test_y_true.npy', 'Results/17DCNN_test_y_proba.npy')
global_precision.append(precision)
global_recall.append(recall)

precision, recall = get_precision_recall('Results/DeepBCTPred_y_true.npy', 'Results/DeepBCTPred_y_proba.npy')
global_precision.append(precision)
global_recall.append(recall)

model_mapper = {
    0: "ResNet_18+SVM",
    1: "ViT",
    2: "AlexNet",
    3: "17DCNN",
    4: "DeepBCTPred"
}

class_mapper = {
    0: "HGC",
    1: "LGC",
    2: "NST",
    3: "NTL"
}

# Plot the macro Precision-Recall curve
plt.figure(figsize=(6, 5))
for j in range(len(global_precision)):
    interp_recall, interp_precision = get_interpolated_scores(global_recall[j], global_precision[j])
    plt.plot(interp_recall, interp_precision, linewidth=2, label=f'{model_mapper[j]}')

plt.xlabel('Recall', labelpad=5)
plt.ylabel('Precision', labelpad=5)
plt.title(f'Macro-Average Precision-Recall Curve')
plt.legend(loc='best', fancybox=True, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()