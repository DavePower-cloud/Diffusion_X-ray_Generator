from __future__ import annotations

import math

import tensorflow as tf
from keras.saving import register_keras_serializable


def linear_diffusion_schedule(diffusion_times):
    min_rate = 0.0001
    max_rate = 0.02

    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas

    alpha_bars = tf.math.cumprod(alphas, axis=0)
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1.0 - alpha_bars)

    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


def build_schedule_arrays(T: int = 1000):
    diffusion_times = tf.convert_to_tensor([x / T for x in range(T)], dtype=tf.float32)

    linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(diffusion_times)
    cosine_noise_rates, cosine_signal_rates = cosine_diffusion_schedule(diffusion_times)
    offset_noise_rates, offset_signal_rates = offset_cosine_diffusion_schedule(diffusion_times)

    return {
        "diffusion_times": diffusion_times,
        "linear_noise_rates": linear_noise_rates,
        "linear_signal_rates": linear_signal_rates,
        "cosine_noise_rates": cosine_noise_rates,
        "cosine_signal_rates": cosine_signal_rates,
        "offset_noise_rates": offset_noise_rates,
        "offset_signal_rates": offset_signal_rates,
    }


def build_sinusoidal_embedding(noise_embedding_size: int):
    @register_keras_serializable()
    def sinusoidal_embedding(x):
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(1.0),
                tf.math.log(1000.0),
                noise_embedding_size // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [
                tf.sin(angular_speeds * x),
                tf.cos(angular_speeds * x),
            ],
            axis=3,
        )
        return embeddings

    return sinusoidal_embedding
