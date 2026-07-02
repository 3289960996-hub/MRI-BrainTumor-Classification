import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def validate_dataset_dir(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Expected a folder with one subfolder per class."
        )

    class_names = [
        label
        for label in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, label))
    ]
    if not class_names:
        raise ValueError(
            f"No class folders found in: {data_dir}. "
            "Expected subfolders such as glioma, meningioma, notumor, pituitary."
        )

    return class_names


def build_dataframe(data_dir):
    validate_dataset_dir(data_dir)

    records = [
        (label, os.path.join(data_dir, label, image))
        for label in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, label))
        for image in os.listdir(os.path.join(data_dir, label))
        if image.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not records:
        raise ValueError(f"No image files found in dataset directory: {data_dir}")

    classes, class_paths = zip(*records)
    return pd.DataFrame({"Class Path": class_paths, "Class": classes})


def train_df(tr_path):
    return build_dataframe(tr_path)


def test_df(ts_path):
    return build_dataframe(ts_path)


def split_test_valid(ts_df, train_size=0.5, random_state=20):
    valid_df, ts_df = train_test_split(
        ts_df,
        train_size=train_size,
        random_state=random_state,
        stratify=ts_df["Class"],
    )
    return valid_df, ts_df


def load_datasets(train_dir, test_dir, valid_train_size=0.5, random_state=20):
    tr_df = train_df(train_dir)
    ts_df = test_df(test_dir)
    valid_df, ts_df = split_test_valid(
        ts_df,
        train_size=valid_train_size,
        random_state=random_state,
    )
    return tr_df, valid_df, ts_df


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
