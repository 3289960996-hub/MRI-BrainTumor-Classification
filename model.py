import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax


def build_model(img_shape=(299, 299, 3), num_classes=4):
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling="max",
    )

    # for layer in base_model.layers:
    #     layer.trainable = False

    model = Sequential(
        [
            base_model,
            Flatten(),
            Dropout(rate=0.3),
            Dense(128, activation="relu"),
            Dropout(rate=0.25),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()],
    )

    return model
