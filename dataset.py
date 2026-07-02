import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_df(tr_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(tr_path, label, image))
            for label in os.listdir(tr_path)
            if os.path.isdir(os.path.join(tr_path, label))
            for image in os.listdir(os.path.join(tr_path, label))
        ]
    )

    tr_df = pd.DataFrame({"Class Path": class_paths, "Class": classes})
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(ts_path, label, image))
            for label in os.listdir(ts_path)
            if os.path.isdir(os.path.join(ts_path, label))
            for image in os.listdir(os.path.join(ts_path, label))
        ]
    )

    ts_df = pd.DataFrame({"Class Path": class_paths, "Class": classes})
    return ts_df


def split_test_valid(ts_df, train_size=0.5, random_state=20):
    valid_df, ts_df = train_test_split(
        ts_df,
        train_size=train_size,
        random_state=random_state,
        stratify=ts_df["Class"],
    )
    return valid_df, ts_df


def create_generators(
    tr_df,
    valid_df,
    ts_df,
    batch_size=32,
    img_size=(299, 299),
    test_batch_size=16,
):
    _gen = ImageDataGenerator(
        rescale=1 / 255,
        brightness_range=(0.8, 1.2),
    )

    ts_gen = ImageDataGenerator(rescale=1 / 255)

    tr_gen = _gen.flow_from_dataframe(
        tr_df,
        x_col="Class Path",
        y_col="Class",
        batch_size=batch_size,
        target_size=img_size,
    )

    valid_gen = _gen.flow_from_dataframe(
        valid_df,
        x_col="Class Path",
        y_col="Class",
        batch_size=batch_size,
        target_size=img_size,
    )

    ts_gen = ts_gen.flow_from_dataframe(
        ts_df,
        x_col="Class Path",
        y_col="Class",
        batch_size=test_batch_size,
        target_size=img_size,
        shuffle=False,
    )

    return tr_gen, valid_gen, ts_gen
