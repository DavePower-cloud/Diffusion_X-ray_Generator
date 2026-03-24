from __future__ import annotations

import numpy as np

from config import get_default_config
from diffusion import build_sinusoidal_embedding
from model import DiffusionModel, build_unet
from utils import display, ensure_dir, set_seed


def main():
    config = get_default_config()

    ensure_dir(config.output_dir)
    ensure_dir(config.checkpoint_dir)

    set_seed(config.seed)

    sinusoidal_embedding = build_sinusoidal_embedding(config.noise_embedding_size)
    unet = build_unet(config.image_size, config.noise_embedding_size, sinusoidal_embedding)

    ddm = DiffusionModel(unet=unet, ema=config.ema)
    ddm.built = True
    ddm.load_weights(config.checkpoint_dir / "checkpoint.weights.h5")

    generated_images = ddm.generate(
        num_images=10,
        diffusion_steps=config.plot_diffusion_steps,
        image_size=config.image_size,
    ).numpy()

    display(
        generated_images,
        save_to=config.output_dir / "generated_samples.png",
    )

    for diffusion_steps in list(np.arange(1, 6, 1)) + [20] + [100]:
        set_seed(config.seed)

        generated_images = ddm.generate(
            num_images=10,
            diffusion_steps=int(diffusion_steps),
            image_size=config.image_size,
        ).numpy()

        display(
            generated_images,
            n=10,
            size=(20, 3),
            save_to=config.output_dir / f"xray_diffusion_steps_{int(diffusion_steps):03d}.png",
        )


if __name__ == "__main__":
    main()
