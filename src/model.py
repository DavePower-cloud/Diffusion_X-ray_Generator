from __future__ import annotations

from tensorflow.keras import activations, layers, losses, metrics, models

from diffusion import offset_cosine_diffusion_schedule


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)

        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width,
            kernel_size=3,
            padding="same",
            activation=activations.swish,
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])

        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def build_unet(image_size: int, noise_embedding_size: int, sinusoidal_embedding):
    noisy_images = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
    noise_embedding = layers.UpSampling2D(size=image_size, interpolation="nearest")(noise_embedding)

    x = layers.Concatenate()([x, noise_embedding])

    skips = []

    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])
    x = DownBlock(96, block_depth=2)([x, skips])

    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)

    x = UpBlock(96, block_depth=2)([x, skips])
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return models.Model([noisy_images, noise_variances], x, name="unet")


class DiffusionModel(models.Model):
    def __init__(self, unet, ema: float = 0.999):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = unet
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule
        self.ema = ema

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return images.clip(0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = layers.Lambda(
                lambda _: 1.0 - step * step_size
            )(initial_noise)

            diffusion_times = diffusion_times[:, :1, :1, :1]
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self.denoise(
                current_images,
                noise_rates,
                signal_rates,
                training=False,
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)

            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps, image_size, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, image_size, image_size, 3)
            )

        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=tf.shape(images))

        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1),
            minval=0.0,
            maxval=1.0,
        )

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, _ = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=True,
            )
            noise_loss = self.compiled_loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {"n_loss": self.noise_loss_tracker.result()}
