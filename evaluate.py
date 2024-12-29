import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from data_loader import get_dataloader
from model import ResNetModel
import argparse

"""
python evaluate.py --model_name resnet50 --data_dir TLDataset/ALL/PKG-C-NMC2019/C-NMC_test_prelim_phase_data --model_path model_checkpoints/ALL/resnet50_best.pth --num_classes 2 --batch_size 32
"""

def evaluate_model(model, data_loader, num_classes, device):
    """Evaluate the model and compute performance metrics."""
    model.eval()
    all_preds, all_labels = [], []

    model.to(device)

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Compute overall metrics
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n--- Overall Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Binarize labels for PR and ROC curves
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    all_preds_bin = label_binarize(all_preds, classes=range(num_classes))

    # Compute Precision, Recall, and ROC-AUC for each class
    precision_class, recall_class, roc_auc = {}, {}, {}
    for i in range(num_classes):
        precision_class[i], recall_class[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        roc_auc[i] = auc(fpr, tpr)

    return cm, precision_class, recall_class, roc_auc


def plot_confusion_matrix(cm, classes):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate cells with values
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


def plot_curves(metric, x, y, class_idx, curve_type):
    """Generic function to plot Precision-Recall or ROC curves."""
    plt.figure()
    plt.plot(x, y, label=f'Class {class_idx}')
    plt.xlabel('Recall' if curve_type == 'PR' else 'False Positive Rate')
    plt.ylabel('Precision' if curve_type == 'PR' else 'True Positive Rate')
    plt.title(f'{curve_type} curve for class {class_idx}')
    plt.legend(loc="best")
    plt.savefig(f'class_{class_idx}_{curve_type}_curve.png')  # Save each curve
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ResNet Model')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model name')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, shuffle=False)

    # Initialize and load model
    model = ResNetModel(model_name=args.model_name, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    cm, precision_class, recall_class, roc_auc = evaluate_model(model, test_loader, args.num_classes, device)

    plot_confusion_matrix(cm, classes=[str(i) for i in range(args.num_classes)])

    for i in range(args.num_classes):
        # Precision-Recall Curve
        plot_curves('PR', recall_class[i], precision_class[i], class_idx=i, curve_type='PR')
        # ROC Curve
        plot_curves('ROC', recall_class[i], precision_class[i], class_idx=i, curve_type='ROC')

    print("Evaluation Complete!")
