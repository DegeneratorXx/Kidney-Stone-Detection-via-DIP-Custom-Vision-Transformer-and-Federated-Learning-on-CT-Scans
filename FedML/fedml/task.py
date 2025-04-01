"""FedML: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from timm import create_model

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

def load_pretrained_model(model_path=r"../../src/model20.pth"):
    """Load the pre-trained model."""
    net = Net(input_shape=(224, 224, 3), num_classes=2)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move to GPU if available
    return net

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
