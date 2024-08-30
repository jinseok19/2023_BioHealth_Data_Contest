import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
from tqdm import tqdm


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
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


# submission
class BaselineTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df['filename'].tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_fname = f'/DATA/test/images/{img_name}'
        img = Image.open(img_fname)
        
        if self.transform:
            img = self.transform(img)
        
        return img


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# hyperparameters
num_batches = 32

test_df = pd.read_csv(f'/DATA/test/test.csv')
test_dataset = BaselineTestDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=num_batches, shuffle=False)

model = BaselineModel().to(device)
model.load_state_dict(torch.load('/USER/RESULT/FT_RMSdragon.pth'))

model.eval()
preds_list = []
with torch.no_grad():
    for image in tqdm(test_loader):
        image = image.to(device)
        outputs = model(image).view(-1)
        
        preds = torch.sigmoid(outputs).round()
        preds_list += preds.cpu().numpy().tolist()

test_df['risk'] = preds_list
test_df['risk'] = test_df['risk'].apply(lambda x: 'high' if x == 1 else 'low')
test_df.to_csv('/USER/RESULT/fin_tuned_model_AdamWRMSdragon2.csv', index=False)
