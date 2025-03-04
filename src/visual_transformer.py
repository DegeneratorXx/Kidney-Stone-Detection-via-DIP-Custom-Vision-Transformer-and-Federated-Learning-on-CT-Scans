import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub

class ConvNeXtViT(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ConvNeXtViT, self).__init__()
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.embed_dim = 768
        self.num_heads = 8
        self.ff_dim = 512

        # ConvNeXt Backbone
        self.convnext = self.build_convnext()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Transformer Encoder
        self.transformer = self.build_transformer()
        self.reshape = layers.Reshape((1, self.embed_dim))
        self.global_avg_pool_1d = layers.GlobalAveragePooling1D()
        
        # Classification Head
        self.layer_norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def build_convnext(self):
        base_model = hub.KerasLayer("https://tfhub.dev/sayakpaul/convnext_tiny_imagenet/1", trainable=True)
        inputs = layers.Input(shape=self.input_shape_)
        x = base_model(inputs)

        # Wrap the base_model into a Functional API model
        model = models.Model(inputs, x, name="ConvNeXt_Backbone")

        # Freeze 80% of layers
        num_layers = len(model.layers)
        freeze_upto = int(num_layers * 0.8)  # 80% layers
        print(f"Freezing {freeze_upto}/{num_layers} layers")

        for layer in model.layers[:freeze_upto]:
            layer.trainable = False

        return model


    def build_transformer(self):
        att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        ffn = models.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dense(self.embed_dim),
        ])
        return att, ffn

    def call(self, inputs):
        # ConvNeXt Feature Extraction
        features = self.convnext(inputs)
        features = self.global_avg_pool(features)
        patches = self.reshape(features)

        # Transformer Encoder
        att, ffn = self.transformer
        attn_output = att(patches, patches)
        out1 = layers.LayerNormalization(epsilon=1e-6)(patches + attn_output)
        ffn_output = ffn(out1)
        encoded = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        encoded = self.global_avg_pool_1d(encoded)

        # Classification Head
        x = self.layer_norm(encoded)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.classifier(x)

# Compile and Test
# input_shape = (224, 224, 3)
# num_classes = 2
# model = ConvNeXtViT(input_shape, num_classes)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.build(input_shape=(None, *input_shape))
# model.summary()