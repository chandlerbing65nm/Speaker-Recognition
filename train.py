import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from joblib import dump
import argparse
import torchaudio.transforms as T

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import random

def save_confusion_matrix(val_targets, val_preds, model_type, save_path='Plots/'):
    # Compute confusion matrix
    cm = confusion_matrix(val_targets, val_preds)
    
    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the plot
    plt.savefig(f'{save_path}{model_type}_confusion_matrix.png')

# Dataset Class
class SpeakerAudioDataset(Dataset):
    def __init__(self, root_dir, max_length=1373, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Timm', 'Shenna', 'John']
        self.data = []
        
        # MFCC transformation
        self.mfcc_transform = T.MFCC(sample_rate=16000)
        self.pad_length = max_length

        # Now iterate again to create the dataset with the computed pad length
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for file_name in tqdm(os.listdir(class_path), f'looping thru {class_name} samples'):
                if file_name.endswith(('.mp4')):
                    file_path = os.path.join(class_path, file_name)

                    # Load audio
                    waveform, sample_rate = torchaudio.load(file_path)

                    # Apply MFCC
                    mfcc_features = self.mfcc_transform(waveform).squeeze(0)
                    # mfcc_features = self.mfcc_transform(waveform).squeeze(0)

                    # If the audio file has multiple channels, average them
                    if len(mfcc_features.shape) == 3:
                        mfcc_features = torch.mean(mfcc_features, dim=0)

                    # Pad to the same length
                    if mfcc_features.shape[-1] < self.pad_length:
                        mfcc_features = torch.nn.functional.pad(mfcc_features, (0, self.pad_length - mfcc_features.shape[-1]))

                    # Append a tuple (mfcc_features, class_index)
                    self.data.append((mfcc_features, self.classes.index(class_name)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Vanilla CNN Model
class VanillaCNN(nn.Module):
    def __init__(self, num_classes, num_filters=64):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=num_filters, kernel_size=3)
        self.relu = nn.ReLU()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_avg_pooling(x)
        x = x.squeeze(-1) # Remove spatial dimension [batch, num_filters]
        x = self.fc(x)
        return x

# Main function
def main(root_dir, model_type='CNN', batch_size=2, num_classes=3, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    dataset = SpeakerAudioDataset(root_dir+'/Train') # 70%
    # val_dataset = SpeakerAudioDataset(root_dir+'/Val') # 30%

    # Extract labels from the dataset
    labels = [label for _, label in dataset]

    # Perform a stratified shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(range(len(dataset)), labels))

    # Convert to PyTorch Datasets if necessary
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    # train_size = int(0.7 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        data, labels = zip(*batch)
        data = torch.stack(data)  # No need to call torch.tensor on already constructed tensors
        labels = torch.tensor(labels)
        return data, labels

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if model_type == 'CNN':
        # Train CNN
        model = VanillaCNN(num_classes=num_classes)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

        # Early stopping parameters
        patience = 5
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(epochs): # Number of epochs
            print('\n')
            print(f'Epoch {epoch+1}')
            print('-' * 10)
            model.train()
            train_loss = 0
            train_preds, train_targets = [], []
            for samples, labels in tqdm(train_loader, 'Training'):
                samples, labels = samples.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                train_preds.extend(predicted.tolist())
                train_targets.extend(labels.tolist())

            train_acc = accuracy_score(train_targets, train_preds)
            train_precision = precision_score(train_targets, train_preds, average='macro', zero_division=0)
            train_recall = recall_score(train_targets, train_preds, average='macro')
            train_f1 = f1_score(train_targets, train_preds, average='macro')

            # Validate
            model.eval()
            val_loss = 0
            val_preds, val_targets = [], []
            for samples, labels in tqdm(val_loader, 'Validation'):
                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                val_preds.extend(predicted.tolist())
                val_targets.extend(labels.tolist())

            save_confusion_matrix(val_targets, val_preds, model_type='CNN')

            val_loss_avg = val_loss / len(val_loader)
            val_acc = accuracy_score(val_targets, val_preds)
            val_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
            val_recall = recall_score(val_targets, val_preds, average='macro')
            val_f1 = f1_score(val_targets, val_preds, average='macro')

            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}\n")

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                counter = 0
                torch.save(model.state_dict(), f'Checkpoints/{model_type}_best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    elif model_type == 'SVM':
        # Prepare data for SVM
        X_train = [sample.numpy().flatten() for sample, _ in train_dataset]
        y_train = [label for _, label in train_dataset]
        X_val = [sample.numpy().flatten() for sample, _ in val_dataset]
        y_val = [label for _, label in val_dataset]

        # Scale the data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # Train SVM
        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train)

        # Save the scaler
        scaler_path = f'Checkpoints/{model_type}_scaler.joblib'
        dump(scaler, scaler_path)

        # Compute metrics for training data
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='macro')
        train_f1 = f1_score(y_train, y_train_pred, average='macro')

        # Validate
        y_val_pred = clf.predict(X_val)
        save_confusion_matrix(y_val, y_val_pred, model_type='SVM')

        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')

        print(f"Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}\n")

        # Save the model    
        dump(clf, f'Checkpoints/{model_type}_best_model.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for speaker audio recognition")
    parser.add_argument("--root_dir", type=str, help="Path to the root directory containing the data")
    parser.add_argument("--model_type", type=str, default="SVM", choices=["CNN", "SVM"], help="Model type to train ('CNN' or 'SVM')")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes in the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()

    args.root_dir = './Dataset'

    main(
        root_dir=args.root_dir, 
        model_type=args.model_type, 
        batch_size=args.batch_size, 
        num_classes=args.num_classes, 
        epochs=args.epochs)




