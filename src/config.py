from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    dataset_path: Path
    checkpoint_dir: Path
    output_dir: Path
    log_dir: Path
    image_size: int
    batch_size: int
    dataset_repetitions: int
    load_model: bool
    noise_embedding_size: int
    plot_diffusion_steps: int
    ema: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seed: int


def get_default_config() -> AppConfig:
    return AppConfig(
        dataset_path=Path("data/chest_xray/train/NORMAL"),
        checkpoint_dir=Path("checkpoint"),
        output_dir=Path("outputs"),
        log_dir=Path("logs"),
        image_size=64,
        batch_size=64,
        dataset_repetitions=5,
        load_model=False,
        noise_embedding_size=32,
        plot_diffusion_steps=20,
        ema=0.999,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=100,
        seed=42,
    )
