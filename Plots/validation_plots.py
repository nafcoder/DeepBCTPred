import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12.5})

# Load the data
df = pd.read_csv('Results/model_validation.csv')

train_losses = df['Train_Loss'].values
val_losses = df['Val_Loss'].values

train_accuracies = df['Train_Accuracy'].values
val_accuracies = df['Val_Accuracy'].values

train_f1_scores = df['Train_F1'].values
val_f1_scores = df['Val_F1'].values

epochs = df['Epoch'].values

# Plot the loss
plt.figure(figsize=(6, 5))
plt.plot(epochs, train_losses, marker='o', color='royalblue', linewidth=2, markersize=3, label='Training Loss')
plt.plot(epochs, val_losses, marker='s', color='orange', linewidth=2, markersize=3, label='Validation Loss')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.6)

# Add labels and title
plt.xlabel('Epochs', labelpad=10)
plt.ylabel('Loss', labelpad=10)
plt.title('Training and Validation Loss vs Epochs', fontweight='bold', pad=15)

# Customize legend
plt.legend(loc='upper right', fancybox=True, shadow=True, borderpad=1)

# Adjust axes and ticks
plt.tight_layout()

# Show plot
plt.show()

# Plot the accuracy
plt.figure(figsize=(6, 5))
plt.plot(epochs, train_accuracies, marker='o', color='royalblue', linewidth=2, markersize=3, label='Training Accuracy')
plt.plot(epochs, val_accuracies, marker='s', color='orange', linewidth=2, markersize=3, label='Validation Accuracy')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.6)

# Add labels and title
plt.xlabel('Epochs', labelpad=10)
plt.ylabel('Accuracy', labelpad=10)
plt.title('Training and Validation Accuracy vs Epochs', fontweight='bold', pad=15)

# Customize legend
plt.legend(loc='lower right', fancybox=True, shadow=True, borderpad=1)

# Adjust axes and ticks
plt.tight_layout()

# Show plot
plt.show()

# Plot the F1 score
plt.figure(figsize=(6, 5))
plt.plot(epochs, train_f1_scores, marker='o', color='royalblue', linewidth=2, markersize=3, label='Training F1 Score')
plt.plot(epochs, val_f1_scores, marker='s', color='orange', linewidth=2, markersize=3, label='Validation F1 Score')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.6)

# Add labels and title
plt.xlabel('Epochs', labelpad=10)
plt.ylabel('F1 Score', labelpad=10)
plt.title('Training and Validation F1 Score vs Epochs', fontweight='bold', pad=15)

# Customize legend
plt.legend(loc='lower right', fancybox=True, shadow=True, borderpad=1)

# Adjust axes and ticks
plt.tight_layout()

# Show plot
plt.show()
