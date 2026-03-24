from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def sample_batch(dataset):
    return dataset.take(1).get_single_element()


def display(
    images,
    n: int = 10,
    size=(20, 3),
    cmap: str = "gray_r",
    as_type: str = "float32",
    save_to: Optional[Path] = None,
):
    if images is None:
        raise ValueError("No images provided to display().")

    images = np.array(images)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    if as_type is not None:
        images = images.astype(as_type)

    plt.figure(figsize=size)

    for i in range(min(n, len(images))):
        plt.subplot(1, min(n, len(images)), i + 1)
        image = images[i]

        if image.shape[-1] == 1:
            image = image[..., 0]

        plt.imshow(image, cmap=cmap)
        plt.axis("off")

    plt.tight_layout()

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight", dpi=200)

    plt.show()
