import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
plt.rcParams.update({'font.size': 12.5})


def get_roc(y_true_path, y_scores_path, n_classes=4):
    y_true = np.load(y_true_path)
    y_scores = np.load(y_scores_path)

    # Binarize the true labels for multiclass ROC curve computation
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Calculate ROC and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def get_interpolated_roc(fpr, tpr, n_classes=4):
    # Compute macro-averaged ROC curve
    all_tprs = []

    for i in range(n_classes):
        # Interpolate TPR for a consistent FPR grid
        interp_fpr = np.linspace(-1e-16, 1, 100)  # Standard FPR grid
        interp_tpr = np.interp(interp_fpr, fpr[i], tpr[i])
        all_tprs.append(interp_tpr)

    # Macro-average of TPRs across classes
    macro_tpr = np.mean(all_tprs, axis=0)

    return interp_fpr, macro_tpr


# Simulate some data
n_classes = 4
global_fpr = []
global_tpr = []
global_auc = []

fpr, tpr, roc_auc = get_roc('Results/ResNet_18+SVM_y_true.npy', 'Results/ResNet_18+SVM_y_proba.npy')
global_fpr.append(fpr)
global_tpr.append(tpr)
global_auc.append(roc_auc)

fpr, tpr, roc_auc = get_roc('Results/ViT_y_true.npy', 'Results/ViT_y_proba.npy')
global_fpr.append(fpr)
global_tpr.append(tpr)
global_auc.append(roc_auc)

fpr, tpr, roc_auc = get_roc('Results/AlexNet_test_y_true.npy', 'Results/AlexNet_test_y_proba.npy')
global_fpr.append(fpr)
global_tpr.append(tpr)
global_auc.append(roc_auc)

fpr, tpr, roc_auc = get_roc('Results/17DCNN_test_y_true.npy', 'Results/17DCNN_test_y_proba.npy')
global_fpr.append(fpr)
global_tpr.append(tpr)
global_auc.append(roc_auc)

fpr, tpr, roc_auc = get_roc('Results/DeepBCTPred_y_true.npy', 'Results/DeepBCTPred_y_proba.npy')
global_fpr.append(fpr)
global_tpr.append(tpr)
global_auc.append(roc_auc)

model_mapper = {
    0: "ResNet_18+SVM",
    1: "ViT",
    2: "AlexNet",
    3: "17DCNN",
    4: "DeepBCTPred"
}

# Plot the macro ROC curve
plt.figure(figsize=(6, 5))
for j in range(len(global_fpr)):
    interp_fpr, interp_tpr = get_interpolated_roc(global_fpr[j], global_tpr[j])
    plt.plot(interp_fpr, interp_tpr, linewidth=2, label=f'{model_mapper[j]}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)', labelpad=5)
plt.ylabel('True Positive Rate (TPR)', labelpad=5)
plt.title(f'Macro-Average ROC Curve')
plt.legend(loc='best', fancybox=True, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
