#!/usr/bin/env python3.6

import tensorflow.keras as krs
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model


def main():
    image_shape = (28, 28)
    latent_space_shape = (100,)
    optimizer = krs.optimizers.Adam(lr=0.0002, beta_1=0.5)

    # Build the discriminator network
    discriminator = build_discriminator(input_shape=image_shape)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Build the generator network
    discriminator.trainable = False  # freeze discriminator weights after compilation
    generator = build_generator(input_shape=latent_space_shape, output_shape=image_shape)

    # Build the adversarial network
    GAN = GenerativeAdversarialNetwork(generator, discriminator, latent_space_shape, optimizer)

    # Load the dataset and preprocess
    (mnist_images, _), (_, _) = krs.datasets.mnist.load_data()
    mnist_images = 2.0 * mnist_images / 255.0 - 1.0  # between -1 and 1

    # Training the adversarial network
    GAN.train(mnist_images, epochs=30000, batch_size=32, save_interval=100)

    # Show a sample
    noise = GAN.create_noise_samples(1)
    sample = generator.predict(noise)
    plt.imshow(sample[0, :, :], cmap='gray')
    plt.show()


def build_discriminator(input_shape):
    model = krs.Sequential([
        krs.layers.Flatten(input_shape=input_shape),
        krs.layers.Dense(512),
        krs.layers.LeakyReLU(alpha=0.2),
        krs.layers.Dense(256),
        krs.layers.LeakyReLU(alpha=0.2),
        krs.layers.Dense(1, activation='sigmoid'),
    ], name='Discriminator')
    print('\tDiscriminator network')
    model.summary()
    return model


def build_generator(input_shape, output_shape):
    flat_dims = np.prod(output_shape)
    model = krs.Sequential([
        krs.layers.Dense(256, input_shape=input_shape),
        krs.layers.LeakyReLU(alpha=0.2),
        krs.layers.BatchNormalization(momentum=0.8),
        krs.layers.Dense(512),
        krs.layers.LeakyReLU(alpha=0.2),
        krs.layers.BatchNormalization(momentum=0.8),
        krs.layers.Dense(1024),
        krs.layers.LeakyReLU(alpha=0.2),
        krs.layers.BatchNormalization(momentum=0.8),
        krs.layers.Dense(flat_dims, activation='tanh'),
        krs.layers.Reshape(output_shape),
    ], name='Generator')
    print('\tGenerator network')
    model.summary()
    return model


class GenerativeAdversarialNetwork:
    def __init__(self, generator, discriminator, latent_space_shape, optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_shape = latent_space_shape
        # Model flow
        noise_layer = krs.Input(shape=self.latent_shape, name='Noice')
        generated_data = self.generator(noise_layer)
        real_or_not = self.discriminator(generated_data)
        self.model = krs.Model(inputs=[noise_layer], outputs=[real_or_not], name='Adversarial')
        # Summary
        print('\tAdversarial network')
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
        plot_model(self.model, to_file='images/adversarial-model.png', show_shapes=True, show_layer_names=True)
        plot_model(self.generator, to_file='images/generator-model.png', show_shapes=True, show_layer_names=True)
        plot_model(self.discriminator, to_file='images/discriminator-model.png', show_shapes=True, show_layer_names=True)

    def create_noise_samples(self, vector_size):
        return np.random.normal(0, 1, (vector_size, self.latent_shape[0]))

    def train(self, data, epochs, batch_size=128, save_interval=100):
        # Real/fake vector labels
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select a random batch of real images
            random_indices = np.random.randint(0, data.shape[0], batch_size)
            real_images = data[random_indices]

            # Generate a batch of fake images
            noise = self.create_noise_samples(batch_size)
            fake_images = self.generator.predict(noise)

            # Train the discriminator what is real and what is fake
            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            d_loss, d_accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Strive to learn generating images that the discriminator thinks are `real`
            noise = self.create_noise_samples(batch_size)
            a_loss = self.model.train_on_batch(noise, real)

            print(f'Epoch: {epoch}, D cost: {d_loss:.6f}, acc.: {100 * d_accuracy:.2f}%, A cost: {a_loss}]')
            if epoch % save_interval == 0:
                self.create_sample_image(epoch)
        self.create_sample_image('final')

    def create_sample_image(self, suffix):
        rows, columns = 5, 5
        noise = self.create_noise_samples(rows * columns)
        generated_images = self.generator.predict(noise)

        # Rescale between the range of (0, 1)
        generated_images = 0.5 * (generated_images + 1)
        figure, axes = plt.subplots(rows, columns)
        for row in range(rows):
            for col in range(columns):
                image = generated_images[row + rows * col, :, :]
                axes[row, col].imshow(image, cmap='gray')
                axes[row, col].axis('off')
        figure.savefig(f'images/{suffix}.png')
        plt.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

