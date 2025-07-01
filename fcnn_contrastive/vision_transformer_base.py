import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import regularizers
from fourier_transform_layer.fourier_transform_layer import FTL, FTLSuperResolution


class MultiHeadFourierAttention(nn.Layer):
    def __init__(
            self,
            embed_dim: tuple,
            num_heads: int=8
        ):
        super(MultiHeadFourierAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = np.prod(embed_dim) * num_heads

        self.query = nn.Dense(self.projection_dim)
        self.key = nn.Dense(self.projection_dim)
        self.value = nn.Dense(self.projection_dim)
        self.combine_heads = nn.Dense(self.projection_dim)

    def attention(
            self,
            query,
            key,
            value
        ):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.math.reduce_prod(tf.shape(key)[1:]), tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(
            self,
            x,
            batch_size
        ):
        x = tf.reshape(
            x, (-1, *self.embed_dim, 1)
        )
        x = FTL(
            activation=None,
            kernel_initializer="ones",
            #kernel_regularizer=regularizers.l1(1e-7),
            train_imaginary=False,
            inverse=False,
            calculate_abs=True,
            normalize_to_image_shape=False,
            already_fft=False,
            use_bias=False,
            bias_initializer="he_normal",
            #bias_regularizer=regularizers.l1_l2(1e-7),
        )(x)
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim // self.num_heads)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
            self,
            inputs
        ):
        batch_size = tf.shape(inputs)[0]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.projection_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Dense(embed_dim)
        self.key_dense = nn.Dense(embed_dim)
        self.value_dense = nn.Dense(embed_dim)
        self.combine_heads = nn.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        # x = tf.reshape(
        #     x, (-1, *[np.int32(np.sqrt(self.embed_dim))] * 2, 1)
        # )
        # x = FTL(
        #     activation="selu",
        #     kernel_initializer="ones",
        #     kernel_regularizer=regularizers.l1(1e-7),
        #     train_imaginary=True,
        #     inverse=False,
        #     calculate_abs=True,
        #     normalize_to_image_shape=False,
        #     already_fft=False,
        #     use_bias=True,
        #     bias_initializer="he_normal",
        #     #bias_regularizer=regularizers.l1_l2(1e-7),
        # )(x)
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                nn.Dense(mlp_dim, activation=tfa.activations.gelu),
                nn.Dropout(dropout),
                nn.Dense(embed_dim),
                nn.Dropout(dropout),
            ]
        )
        self.layernorm1 = nn.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = nn.LayerNormalization(epsilon=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        #self.rescale = Rescaling(1.0 / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = nn.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                nn.LayerNormalization(epsilon=1e-6),
                nn.Dense(mlp_dim, activation=tfa.activations.gelu),
                nn.Dropout(dropout),
                nn.Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        #x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x


class Patcher(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        channels=1,
    ):
        super(Patcher, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.channels = channels

    def extract_patches(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, self.num_patches * self.channels])
        return patches

    def call(self, input_tensor, **kwargs):
        batch_size = tf.shape(input_tensor)[0]
        patches = self.extract_patches(input_tensor)
        return patches

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.patch_size, self.patch_size, self.num_patches * self.channels]


class Reverser(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        channels=1,
    ):
        super(Reverser, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.axes_1_2_size = int(np.sqrt((image_size ** 2) / (patch_size ** 2)))

        self.patch_size = patch_size
        self.channels = channels
        self.image_size = image_size

    def reverse_patches(self, patches):
        reconstruct = tf.reshape(
            patches,
            (-1, self.axes_1_2_size, self.axes_1_2_size, self.patch_size, self.patch_size, self.channels)
            )
        # Tranpose the axes (I got this axes tuple for transpose via experimentation)
        reconstruct = tf.transpose(reconstruct, (0, 1, 3, 2, 4, 5))
        # Reshape back
        reconstruct = tf.reshape(reconstruct, (-1, self.image_size, self.image_size, self.channels))
        return reconstruct

    def call(self, input_tensor, **kwargs):
        batch_size = tf.shape(input_tensor)[0]
        images = self.reverse_patches(input_tensor)
        return images

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_size, self.image_size, self.channels]


def patching_ftl_block(
        input_tensor,
        patch_size,
        image_size=128,
        channels=1,
        absolute=True,
    ):
    if patch_size is None:
        ftl = input_tensor
    else:
        ftl = Patcher(
            image_size,
            patch_size,
            channels
            )(input_tensor)
    ftl = FTL(
        activation="selu",
        train_imaginary=True,
        inverse=False,
        calculate_abs=absolute,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l1(1e-7),
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=regularizers.l1(1e-4),
        )(ftl)
    if not absolute:
        # real - imaginary pair extraction
        pairs = tf.split(ftl, num_or_size_splits=channels, axis=-1)
        reals, imags = [], []
        for pair in pairs:
            pair_reversed = Reverser(
                image_size,
                patch_size,
                2 - absolute
                )(pair)
            reals.append(pair_reversed[:, :, :, 0:1])
            imags.append(pair_reversed[:, :, :, 1:])
        reals = tf.concat(reals, axis=-1)
        imags = tf.concat(imags, axis=-1)
        if patch_size is not None:
            reals_reversed = Reverser(
                image_size,
                patch_size,
                channels
                )(reals)
            imags_reversed = Reverser(
                image_size,
                patch_size,
                channels
                )(imags)
            return reals_reversed, imags_reversed
        return reals, imags
    if patch_size is not None:
        ftl = Reverser(
            image_size,
            patch_size,
            channels * (2 - absolute)
            )(ftl)
    #ftl = nn.AveragePooling2D()(ftl)
    return ftl


def build_model(
        input_shape,
        noof_classes=10,
        patch_sizes=[4],
        calculate_relations=False
    ):
    inp = nn.Input(input_shape)
    blocks = []
    # min 5 in list if channels == 1
    # determine relation pairs:
    if calculate_relations:
        _relations, relations = [], []
        for p1 in patch_sizes:
            for p2 in patch_sizes:
                if sorted([p1, p2]) in _relations:
                    continue
                _relations.append(sorted([p1, p2]))
                relations.append(p1)
                relations.append(p2)
    else:
        relations = patch_sizes
    for ps in relations:
        blocks.append(patching_ftl_block(
            input_tensor=inp,
            patch_size=ps,
            image_size=input_shape[0],
            channels=input_shape[-1],
            absolute=False
        ))
    outs = []
    for block in blocks:
        arch = tf.expand_dims(tf.math.reduce_max(nn.concatenate(block, axis=-1), axis=-1), axis=-1)
        arch = nn.Conv2D(filters=4, kernel_size=9, strides=1, activation="relu")(arch)
        arch = nn.Flatten()(arch)
        arch = MultiHeadSelfAttention(
            embed_dim=32,
            num_heads=4
            )(arch)
        outs.append(arch)
    arch = tf.concat(outs, axis=-1)
    #arch = nn.Conv2D(filters=noof_classes, kernel_size=1, strides=1, activation="softmax")(arch)
    # out = nn.GlobalAveragePooling2D()(arch)
    #arch = nn.Flatten()(arch)
    arch = tf.reshape(arch, [-1, 32 * len(blocks)])
    out = nn.Dense(noof_classes, activation="sigmoid" if noof_classes == 1 else "softmax")(arch)
    # arch = tf.expand_dims(nn.concatenate(blocks, axis=-1), axis=-1)
    # color channel merging
    #arch = nn.Conv3D(filters=128, kernel_size=3, strides=1, activation="relu")(arch)
    # closest relations merging
    #arch = nn.Conv3D(filters=64, kernel_size=3, strides=1, activation="relu")(arch)
    # larger relations merging
    #arch = nn.Conv3D(filters=64, kernel_size=5, strides=1, activation="relu")(arch)
    # classification
    #arch = nn.Conv3D(filters=noof_classes, kernel_size=3, strides=1, activation="sigmoid")(arch)
    # TODO: merge filters and "channels"
    # arch = nn.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu")(arch)
    # arch = nn.Conv2D(filters=128, kernel_size=5, strides=1, activation="relu")(arch)
    # arch = nn.Conv2D(filters=noof_classes, kernel_size=1, strides=1, activation="softmax")(arch)
    # out = nn.GlobalAveragePooling2D()(arch)
    #out = nn.GlobalAveragePooling3D()(arch)
    #flat = nn.Flatten()(attention)
    #out = nn.Dense(noof_classes, activation="softmax")(flat)
    return tf.keras.Model(inp, out)





if __name__ == "__main__":
    # following https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches
    # image = tf.constant([[[1],   [2],  [3],  [4]],
    #                  [[5],   [6],  [7],  [8]],
    #                  [[9],  [10], [11],  [12]],
    #                 [[13], [14], [15],  [16]]])
    #
    # patch_size = [1,2,2,1]
    # patches = tf.image.extract_patches([image],
    #     patch_size, patch_size, [1, 1, 1, 1], 'VALID')
    # patches = tf.reshape(patches, [4, 2, 2, 1])
    # reconstructed = tf.reshape(patches, [1, 4, 4, 1])
    # rec_new = tf.nn.space_to_depth(reconstructed,2)
    # rec_new = tf.reshape(rec_new,[4,4,1])
    #
    # I,P,R_n = [image,patches,rec_new]
    # print(I)
    # print(I.shape)
    # print(P.shape)
    # print(R_n)
    # print(R_n.shape)
    # model = build_model(
    #     input_shape=(64, 64, 1),
    #     noof_classes=2,
    #     patch_sizes=[*[16]*3],
    #     calculate_relations=False,
    # )
    # model.summary()
    attention = MultiHeadSelfAttention(
        embed_dim=256,
        num_heads=8,
    )
    x = attention(np.ones((4, 32, 32, 1)))
    transformer = TransformerBlock(
        embed_dim=256,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1
    )
    x = transformer(np.ones((4, 1024, 256)))
    vit = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=1,
        num_classes=1,
        d_model=256,
        num_heads=8,
        mlp_dim=128,
        channels=1,
        dropout=0.1
    )
    x = vit(np.ones((4, 32, 32, 1)))
