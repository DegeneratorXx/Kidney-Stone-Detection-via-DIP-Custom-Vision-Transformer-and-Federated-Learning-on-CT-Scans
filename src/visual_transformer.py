import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
import numpy as np

class CustomHybridModel(tf.keras.Model):
    def __init__(self, num_classes, patch_size=16, embed_dim=768, num_heads=12, num_layers=6):
        super(CustomHybridModel, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # InceptionV3 Backbone
        self.inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.inception.trainable = False
        
        # Convolutional Layer to Map Patches to Embedding Dimension
        self.conv_proj = layers.Conv2D(embed_dim, (patch_size, patch_size), strides=(patch_size, patch_size), padding='valid')
        
        # Positional Encoding
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, 196, embed_dim), initializer='random_normal', trainable=True)
        self.cls_token = self.add_weight("cls_token", shape=(1, 1, embed_dim), initializer='random_normal', trainable=True)

        # Transformer Encoder Layers
        self.transformer_layers = [
            layers.TransformerBlock(embed_dim, num_heads, embed_dim * 4) for _ in range(num_layers)
        ]

        # Classification Head
        self.fc = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])

    def extract_patches(self, features):
        B, H, W, C = features.shape
        patches = self.conv_proj(features)  # (B, H/P, W/P, embed_dim)
        patches = tf.reshape(patches, (B, -1, self.embed_dim))  # (B, Num_Patches, embed_dim)
        return patches

    def call(self, inputs):
        # Step 1: InceptionV3 Backbone
        features = self.inception(inputs)
        
        # Step 2: Flatten into patches
        patches = self.extract_patches(features)

        # Step 3: Add Positional Encoding
        patches += self.pos_embedding
        
        # Step 4: Append CLS Token
        cls_tokens = tf.broadcast_to(self.cls_token, [tf.shape(patches)[0], 1, self.embed_dim])
        patches = tf.concat([cls_tokens, patches], axis=1)

        # Step 5: Transformer Encoder
        for transformer_layer in self.transformer_layers:
            patches = transformer_layer(patches)

        # Step 6: CLS Token for Classification
        cls_output = patches[:, 0, :]
        return self.fc(cls_output)

# Instantiate Model
# model = CustomHybridModel(num_classes=2)
# model.build(input_shape=(None, 224, 224, 3))
# model.summary()
