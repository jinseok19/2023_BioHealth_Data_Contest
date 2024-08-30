import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from torch.optim import RMSprop
from torch.optim import lr_scheduler



class BaselineDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.df['risk'] = self.df['risk'].apply(lambda x: 1 if x == 'high' else 0)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, label = self.df.iloc[idx]
        img_fname = f'/DATA/train/images/{img_name}'
        img = Image.open(img_fname)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs).view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


# Define the validation function
def valid(model, val_loader, criterion, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).view(-1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            preds = torch.sigmoid(outputs).round()
            metrics.append(f1_score(labels.cpu(), preds.cpu(), average='macro'))
    return losses, metrics


# Load the data
df = pd.read_csv(f'/DATA/train/train.csv')

# transformations
entire_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_batches = 32
# Apply entire_transform to the entire dataset
dataset = BaselineDataset(df, transform=entire_transform)

# train / validation split 
# Split the dataset into train and validation sets
# Adjust the split ratio based on your dataset size
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=num_batches, shuffle=False)


# Fine-tuning hyperparameters
fine_tune_epochs = 21
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FineTuningBaselineModel(nn.Module):
    def __init__(self):
        super(FineTuningBaselineModel, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        #freeze
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Classifier unfreeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        n_features = self.model.classifier.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.model.classifier = self.fc

    def forward(self, x):
        x = self.model(x)
        return x

# Load the pre-trained model
model = FineTuningBaselineModel().to(device)
model.load_state_dict(torch.load('/USER/BEST_TL_AdamWdragon.pth'))

# Fine-tune optimizer and scheduler
fine_tune_optimizer = RMSprop(model.parameters(),
                    lr=1e-3,
                    alpha=0.9,
                    eps=1e-8
                    )

# learning rate scheduler
scheduler = lr_scheduler.StepLR(fine_tune_optimizer, step_size=3, gamma=0.1)

# Fine-tune the model
for epoch in range(fine_tune_epochs):
    train_losses = train(model, train_loader, criterion, fine_tune_optimizer, device)
    val_losses, val_metrics = valid(model, val_loader, criterion, device)
    
    scheduler.step()  # Adjust the learning rate based on the scheduler

    print('Fine-tune Epoch {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Metric: {:.4f}'.format(epoch + 1, np.mean(train_losses), np.mean(val_losses), np.mean(val_metrics)))

# Save the fine-tuned model
torch.save(model.state_dict(), '/USER/FT_RMSdragon.pth')
