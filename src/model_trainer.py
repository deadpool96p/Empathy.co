import tensorflow as tf
from tensorflow.keras import layers, Model

class TimeShuffleAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    def build(self, input_shape):
        _, time, channels = input_shape
        self.temporal_dense = layers.Dense(time // self.reduction, activation='relu')
        self.temporal_restore = layers.Dense(time, activation='sigmoid')
        self.channel_dense = layers.Dense(channels // self.reduction, activation='relu')
        self.channel_restore = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)
    def call(self, inputs):
        temporal_att = tf.reduce_mean(inputs, axis=2, keepdims=True)
        temporal_att = tf.squeeze(temporal_att, axis=2)
        temporal_att = self.temporal_dense(temporal_att)
        temporal_att = self.temporal_restore(temporal_att)
        temporal_att = tf.expand_dims(temporal_att, axis=2)
        channel_att = tf.reduce_mean(inputs, axis=1, keepdims=True)
        channel_att = tf.squeeze(channel_att, axis=1)
        channel_att = self.channel_dense(channel_att)
        channel_att = self.channel_restore(channel_att)
        channel_att = tf.expand_dims(channel_att, axis=1)
        attended = inputs * temporal_att * channel_att
        return attended
    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config

class LightweightConvTransformer(layers.Layer):
    def __init__(self, dim, num_heads=4, expansion=4, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.expansion = expansion
        self.drop_path = drop_path
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size=3, padding='same')
        self.pointwise_conv1 = layers.Conv1D(dim * expansion, kernel_size=1)
        self.pointwise_conv2 = layers.Conv1D(dim, kernel_size=1)
        self.layer_norm1 = layers.LayerNormalization()
        self.multi_head_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.layer_norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(dim * expansion, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(dim)
        ])
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=None):
        shortcut = x
        x_conv = self.depthwise_conv(x)
        x_conv = self.pointwise_conv1(x_conv)
        x_conv = tf.nn.gelu(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv, training=training)

        if self.drop_path > 0 and training:
            batch_size = tf.shape(x)[0]
            keep_prob = 1.0 - self.drop_path
            random_tensor = keep_prob + tf.random.uniform((batch_size,))
            binary_mask = tf.floor(random_tensor)
            binary_mask = tf.reshape(binary_mask, (batch_size, 1, 1))
            x_conv = (x_conv * binary_mask) / keep_prob

        x_conv = shortcut + x_conv
        x_norm = self.layer_norm1(x_conv)
        x_att = self.multi_head_att(x_norm, x_norm)
        x_att = self.dropout(x_att, training=training)
        x_att = x_conv + x_att

        x_norm2 = self.layer_norm2(x_att)
        x_ffn = self.ffn(x_norm2)
        x_ffn = self.dropout(x_ffn, training=training)
        x_out = x_att + x_ffn
        return x_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "expansion": self.expansion,
            "drop_path": self.drop_path
        })
        return config

def create_advanced_model(input_dim, num_emotions, total_steps=1000, 
                          num_blocks=3, dense_units=[128, 64], 
                          dropout_rates=[0.4, 0.3], use_attention=True, 
                          drop_path=0.15):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)

    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    if use_attention:
        x = TimeShuffleAttention(reduction=8)(x)

    for i in range(num_blocks):
        block_drop_path = drop_path * (i / max(1, num_blocks-1))
        x = LightweightConvTransformer(dim=128, num_heads=4, expansion=4, drop_path=block_drop_path)(x)

    x = layers.GlobalAveragePooling1D()(x)

    for units, dr in zip(dense_units, dropout_rates):
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)

    outputs = layers.Dense(num_emotions, activation='softmax')(x)
    model = Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-4, decay_steps=total_steps, alpha=1e-2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_simple_model(input_dim, num_emotions=8):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_emotions, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model