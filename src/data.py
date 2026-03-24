from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import utils


def validate_dataset_path(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}. "
            "Please place the chest X-ray images in the expected local folder."
        )


def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


def load_training_dataset(
    dataset_path: Path,
    image_size: int,
    batch_size: int,
    dataset_repetitions: int,
    seed: int = 42,
):
    validate_dataset_path(dataset_path)

    train_data = utils.image_dataset_from_directory(
        dataset_path,
        labels=None,
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=True,
        seed=seed,
        interpolation="bilinear",
    )

    train = train_data.map(lambda x: preprocess(x))
    train = train.repeat(dataset_repetitions)
    train = train.batch(batch_size, drop_remainder=True)

    return train
