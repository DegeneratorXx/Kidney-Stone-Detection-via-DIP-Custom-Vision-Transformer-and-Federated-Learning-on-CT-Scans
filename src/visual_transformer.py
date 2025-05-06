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


class ConvNeXtModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvNeXtModel, self).__init__()
        
        # ConvNeXt Backbone (from timm)
        self.convnext = create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="")
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze 80% of layers
        total_layers = len(list(self.convnext.parameters()))
        freeze_upto = int(total_layers * 0.8)
        for i, param in enumerate(self.convnext.parameters()):
            if i < freeze_upto:
                param.requires_grad = False
        
        # Classification Head
        self.dense1 = nn.Linear(self.convnext.num_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # ConvNeXt Feature Extraction
        features = self.convnext(x)
        features = self.global_avg_pool(features).view(features.shape[0], -1)
        
        # Classification Head
        x = F.relu(self.dense1(features))
        x = self.dropout(x)
        return F.softmax(self.classifier(x), dim=1)
    

class VisionTransformerModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.embed_dim = 768
        self.num_heads = 8
        self.ff_dim = 512
        
        # Patch Embedding Layer (replaces ConvNeXt)
        self.patch_size = 16
        self.num_patches = (input_shape[1] // self.patch_size) * (input_shape[2] // self.patch_size)
        self.patch_embed = nn.Conv2d(input_shape[0], self.embed_dim, 
                                    kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional Embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
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
        # Create patch embeddings
        patches = self.patch_embed(x)  # [B, embed_dim, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)
        patches = patches + self.position_embeddings
        
        # Transformer Encoder
        attn_output, _ = self.attn(patches, patches, patches)
        out1 = self.layer_norm1(patches + attn_output)
        ffn_output = self.ffn(out1)
        encoded = self.layer_norm2(out1 + ffn_output)
        
        # Classification Head (use only class token)
        cls_token = encoded[:, 0]
        x = self.layer_norm(cls_token)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        return F.softmax(self.classifier(x), dim=1)