import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kagglehub
from tqdm import tqdm
from model import AiDetectorCNN

def train_model():
    print("Downloading lightweight CIFAKE dataset (105MB) from Kaggle...")
    # This dataset contains 32x32 images, perfect for lightning-fast training
    dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    print(f"Dataset downloaded to: {dataset_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard CIFAKE augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'test')
    
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    dataloaders = {'train': train_loader, 'test': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'test': len(val_dataset)}
    
    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")

    if not os.path.exists('weights'):
        os.makedirs('weights')
    with open('weights/class_mapping.json', 'w') as f:
        json.dump({str(v): k for k, v in dict(enumerate(class_names)).items()}, f)

    model = AiDetectorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    best_acc = 0.0

    print("Starting Training (College Project Mode)...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'weights/best_model.pth')

    print(f'\nTraining complete! Best Test Acc: {best_acc:4f}')

if __name__ == '__main__':
    train_model()
