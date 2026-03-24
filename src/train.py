from __future__ import annotations

from tensorflow.keras import callbacks, losses, optimizers

from config import get_default_config
from data import load_training_dataset
from diffusion import build_sinusoidal_embedding
from model import DiffusionModel, build_unet
from utils import display, ensure_dir, sample_batch, set_seed


class ImageGenerator(callbacks.Callback):
    def __init__(self, config):
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(
            num_images=10,
            diffusion_steps=self.config.plot_diffusion_steps,
            image_size=self.config.image_size,
        ).numpy()

        display(
            generated_images,
            save_to=self.config.output_dir / f"generated_xray_img_{epoch:03d}.png",
        )


def main():
    config = get_default_config()

    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.output_dir)
    ensure_dir(config.log_dir)

    set_seed(config.seed)

    train = load_training_dataset(
        dataset_path=config.dataset_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        dataset_repetitions=config.dataset_repetitions,
        seed=config.seed,
    )

    train_sample = sample_batch(train)
    display(train_sample)

    sinusoidal_embedding = build_sinusoidal_embedding(config.noise_embedding_size)
    unet = build_unet(config.image_size, config.noise_embedding_size, sinusoidal_embedding)

    ddm = DiffusionModel(unet=unet, ema=config.ema)
    ddm.normalizer.adapt(train)

    if config.load_model:
        ddm.built = True
        ddm.load_weights(config.checkpoint_dir / "checkpoint.weights.h5")

    ddm.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        loss=losses.MeanAbsoluteError(),
    )

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=config.checkpoint_dir / "checkpoint.weights.h5",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )

    tensorboard_callback = callbacks.TensorBoard(log_dir=config.log_dir)
    image_generator_callback = ImageGenerator(config=config)

    ddm.build(input_shape=(None, config.image_size, config.image_size, 3))

    ddm.fit(
        train,
        epochs=config.epochs,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            image_generator_callback,
        ],
    )


if __name__ == "__main__":
    main()
