"""FedML: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm import create_model
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class Net(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Net, self).__init__()
        self.embed_dim = 768
        self.num_heads = 8
        self.ff_dim = 512
        
        # ConvNeXt Backbone (from timm)
        self.convnext = create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="")
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze 80% of layers
        total_layers = len(list(self.convnext.parameters()))
        freeze_upto = int(total_layers * 0.8)
        for i, param in enumerate(self.convnext.parameters()):
            if i < freeze_upto:
                param.requires_grad = False
        
        # Transformer Encoder
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        
        # Classification Head
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dense1 = nn.Linear(self.embed_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # ConvNeXt Feature Extraction
        features = self.convnext(x)
        features = self.global_avg_pool(features).view(features.shape[0], -1)
        patches = features.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Transformer Encoder
        attn_output, _ = self.attn(patches, patches, patches)
        out1 = self.layer_norm1(patches + attn_output)
        ffn_output = self.ffn(out1)
        encoded = self.layer_norm2(out1 + ffn_output)
        encoded = encoded.squeeze(1)  # (B, embed_dim)
        
        # Classification Head
        x = self.layer_norm(encoded)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        return F.softmax(self.classifier(x), dim=1)

def load_pretrained_model(model_path="model20.pth"):
    """Load the pre-trained model."""
    net = Net(input_shape=(224, 224, 3), num_classes=2)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        net.load_state_dict(state_dict)
    net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return net

def load_data(partition_id: int, num_partitions: int, data_dir=r"C:\Users\parth\Downloads\Dataset_fedml"):
    """Load local dataset with Kidney_Stone and Normal classes."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load train and test datasets
    train_dataset = CustomDataset(root_dir=os.path.join(data_dir, "Train"), transform=transform)
    test_dataset = CustomDataset(root_dir=os.path.join(data_dir, "Test"), transform=transform)
    
    # Split data for federated learning (simple approach - split train data into partitions)
    partition_size = len(train_dataset) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(train_dataset)
    
    # Create subset for this partition
    from torch.utils.data import Subset
    indices = list(range(start_idx, end_idx))
    partition_train = Subset(train_dataset, indices)
    
    # Create data loaders
    trainloader = DataLoader(partition_train, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)