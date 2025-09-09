# Accurate-Detection-of-Hair-Diseases
The proposed approach can reliably identify hair disorders. We constructed the classification model using a Lightweight Convolutional Neural Network (LWCNN), which enables the automatic extraction of features in an end-to-end fashion. Compared to other methods that widely used like ResNet-18, and SqueezeNet.
# Hair Disease Classification (Kaggle-ready)
# Deep Learning Approaches for Accurate Detection of Hair Diseases
# Implements: Lightweight CNN (LWCNN), and transfer learning with ResNet-18 & SqueezeNet (PyTorch)
# Assumes dataset organized like: /kaggle/input/hdi/train/<class>/*.jpg and /kaggle/input/hdi/test/<class>/*.jpg

import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models, datasets

# ---------------------------
# Configuration
# ---------------------------
CFG = {
    'data_dir': '/kaggle/input/hdi',  # adjust if dataset path differs
    'train_dir': '/kaggle/input/hdi/train',
    'test_dir': '/kaggle/input/hdi/test',
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'epochs': 25,
    'lr': 1e-3,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 10,
    'model_save_dir': '/kaggle/working/models'
}

os.makedirs(CFG['model_save_dir'], exist_ok=True)

# Reproducibility
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
if CFG['device']=='cuda':
    torch.cuda.manual_seed_all(CFG['seed'])

# ---------------------------
# Data augmentation (Reflection, Translation, Scale) + normalization
# Reflection -> RandomHorizontalFlip
# Translation & Scale -> RandomAffine + RandomResizedCrop
# ---------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(CFG['img_size'], scale=(0.8, 1.0)),  # scale
    transforms.RandomHorizontalFlip(p=0.5),  # reflection
    transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # translation & small scale
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(int(CFG['img_size']*1.1)),
    transforms.CenterCrop(CFG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Datasets & Dataloaders
# ---------------------------
train_dataset = datasets.ImageFolder(CFG['train_dir'], transform=train_transforms)
val_dataset = datasets.ImageFolder(CFG['test_dir'], transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])

class_names = train_dataset.classes
print(f"Classes ({len(class_names)}):", class_names)

# ---------------------------
# Lightweight DCNN (LWCNN) definition
# Five convolutional blocks; each: Conv2d -> BatchNorm -> LeakyReLU -> MaxPool
# Designed to be parameter-efficient.
# ---------------------------
class LWCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, base_filters=32):
        super(LWCNN, self).__init__()
        layers = []
        channels = in_channels
        filters = base_filters
        # 5 conv blocks
        for i in range(5):
            layers.append(nn.Conv2d(channels, filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # light 1x1 conv to reduce parameters occasionally
            if i in [1, 3]:
                layers.append(nn.Conv2d(filters, filters, kernel_size=1, padding=0, bias=False))
                layers.append(nn.BatchNorm2d(filters))
                layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            channels = filters
            filters = min(filters*2, 256)

        self.features = nn.Sequential(*layers)
        # adaptive pooling -> classifier
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(channels, 128, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Helper: initialize model, optimizer, criterion
# ---------------------------
def build_model(model_name='lwcnn', num_classes=CFG['num_classes']):
    if model_name.lower() == 'lwcnn':
        model = LWCNN(num_classes=num_classes)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(pretrained=True)
        # replace final fc
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif model_name.lower() == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    else:
        raise ValueError('Unknown model')
    return model

# ---------------------------
# Training & evaluation loops
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in tqdm(loader, desc='Train', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)

# ---------------------------
# Full train procedure for a given model name
# ---------------------------

def run_training(model_name='lwcnn'):
    print(f"Building model: {model_name}")
    model = build_model(model_name, num_classes=len(class_names))
    model = model.to(CFG['device'])

    # For transfer learning, freeze feature extractor slightly for ResNet18 & SqueezeNet
    if model_name.lower() in ['resnet18', 'squeezenet']:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=CFG['lr'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = defaultdict(list)

    for epoch in range(1, CFG['epochs']+1):
        print(f"\nEpoch {epoch}/{CFG['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CFG['device'])
        val_loss, val_acc, y_true, y_pred = eval_model(model, val_loader, criterion, CFG['device'])

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CFG['model_save_dir'], f"best_{model_name}.pth")
            torch.save({'model_state_dict': model.state_dict(), 'history': history, 'class_names': class_names}, save_path)
            print(f"Saved best model to {save_path}")

    # after training, return history and last y_true/y_pred
    return model, history, (y_true, y_pred)

# ---------------------------
# Run three experiments
# ---------------------------
if __name__ == '__main__':
    print('Device:', CFG['device'])

    # 1) Lightweight CNN
    lw_model, lw_hist, (lw_ytrue, lw_ypred) = run_training('lwcnn')

    # 2) ResNet-18 transfer learning
    res_model, res_hist, (res_ytrue, res_ypred) = run_training('resnet18')

    # 3) SqueezeNet transfer learning
    sq_model, sq_hist, (sq_ytrue, sq_ypred) = run_training('squeezenet')

    # ---------------------------
    # Evaluation summary & saving predictions
    # ---------------------------
    import sklearn.metrics as skm

    def print_report(y_true, y_pred, title='Report'):
        print(f"\n--- {title} ---")
        acc = skm.accuracy_score(y_true, y_pred)
        print('Accuracy:', acc)
        print('Classification report:\n', skm.classification_report(y_true, y_pred, target_names=class_names))
        cm = skm.confusion_matrix(y_true, y_pred)
        print('Confusion matrix:\n', cm)

    print_report(lw_ytrue, lw_ypred, 'LWCNN Eval')
    print_report(res_ytrue, res_ypred, 'ResNet-18 Eval')
    print_report(sq_ytrue, sq_ypred, 'SqueezeNet Eval')

    # Save final history to CSV for plotting/analysis
    pd.DataFrame(lw_hist).to_csv('/kaggle/working/lwcnn_history.csv', index=False)
    pd.DataFrame(res_hist).to_csv('/kaggle/working/resnet18_history.csv', index=False)
    pd.DataFrame(sq_hist).to_csv('/kaggle/working/squeezenet_history.csv', index=False)

    print('\nAll done. Models & history saved to /kaggle/working and /kaggle/working/models')
