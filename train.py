import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import ResNetModel
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from evaluate import plot_confusion_matrix
import os
import argparse


"""
python train.py --model_name resnet50 --data_dir TLDataset/ALL/PKG-C-NMC2019 --num_classes 2 --num_epochs 50 --batch_size 32 --learning_rate 0.001 --momentum 0.9 --pretrained_path None
"""

def train_model(model_name, data_dir, num_classes, num_epochs, batch_size, pretrained_path=None, learning_rate=0.001, momentum=0.9):
    # Track metrics
    train_losses, val_losses, precisions, recalls = [], [], [], []
    best_f1 = 0.0

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data using get_dataloader
    train_loader = get_dataloader(data_dir=os.path.join(data_dir, 'train'), batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_dir=os.path.join(data_dir, 'val'), batch_size=batch_size, shuffle=False)

    print(f"Training Data: {len(train_loader.dataset)} samples")
    print(f"Validation Data: {len(val_loader.dataset)} samples")

    # Initialize model
    if pretrained_path:
        model = ResNetModel(model_name=model_name, num_classes=num_classes, pretrained=True)
        model.load_state_dict(torch.load(pretrained_path))  # Load pretrained weights
    else:
        model = ResNetModel(model_name=model_name, num_classes=num_classes, pretrained=True) # load the training with ImageNet weights 

    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print(f'Epoch: {epoch + 1}/{num_epochs} | Step: {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}')

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}')

        # Calculate metrics
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        precisions.append(precision)
        recalls.append(recall)

        # Save best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = f'./checkpoints/{model_name}_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model at epoch {epoch+1} with F1 score: {best_f1:.4f}')

        # Save the latest model
        latest_model_path = f'./checkpoints/{model_name}_latest.pth'
        torch.save(model.state_dict(), latest_model_path)

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_name}_loss_plot.png')
    plt.close()

    # Plot precision and recall
    plt.figure()
    plt.plot(range(1, num_epochs + 1), precisions, label='Precision')
    plt.plot(range(1, num_epochs + 1), recalls, label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'{model_name}_precision_recall_plot.png')
    plt.close()

    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure()
    plot_confusion_matrix(cm, classes=[str(i) for i in range(num_classes)])
    plt.title('Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet Model')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')

    args = parser.parse_args()

    train_model(args.model_name, args.data_dir, args.num_classes, args.num_epochs, args.batch_size, args.pretrained_path, args.learning_rate, args.momentum)
