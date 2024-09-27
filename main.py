import tensorflow as tf, tf_keras
from official.vision.modeling.backbones import MobileNet

train_dir = "/home/azureuser/data/Places365/train"
val_dir = "/home/azureuser/data/Places365/val"

patch_size = 56
num_patches = 16
batch_size = 32
img_size = 224
num_classes = 365


class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches, [batch_size, -1, self.patch_size, self.patch_size, 3]
        )
        print(patches)
        return patches


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(tf.add(inputs, attn_output))
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(tf.add(out1, ffn_output))

class MobileNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(MobileNetFeatureExtractor, self).__init__()
        self.mobilenet = MobileNet(model_id="MobileNetGobi", filter_size_scale=0.75)

    def call(self, inputs):
        features = self.mobilenet(inputs)
        return features["5"] 

# Building the model
def create_model():
    inputs = tf_keras.Input(shape=(img_size, img_size, 3), batch_size=batch_size)
    x = PatchExtractor(patch_size)(inputs)
    x = tf.keras.layers.TimeDistributed(MobileNetFeatureExtractor())(x)
    x = tf.squeeze(x, axis=[2, 3])
    x = TransformerBlock(embed_dim=80, num_heads=4, ff_dim=160)(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf_keras.Model(inputs=inputs, outputs=outputs)


def train():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model()

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        # Load training dataset
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=True,
        )

        # Load valing dataset
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )

        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

        checkpoint_path = "./checkpoints/model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.keras"
        checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        model.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset,
            callbacks=[checkpoint_callback],
        )
        
        model.save('./final_model.keras') 

        val_loss, val_acc = model.evaluate(val_dataset, verbose=2)
        print(f"\nval accuracy: {val_acc}\nval loss: {val_loss}")


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    train()
