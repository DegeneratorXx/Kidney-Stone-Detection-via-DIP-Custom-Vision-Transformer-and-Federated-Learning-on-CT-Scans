import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm import create_model

class ConvNeXtViT(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvNeXtViT, self).__init__()
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

# Example usage
# model = ConvNeXtViT(input_shape=(224, 224, 3), num_classes=2)
# print(model)
