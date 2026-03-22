import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, classes, output_path, normalize=True):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
