from tensorflow.keras.applications.vgg19 import VGG19
import math
import tensorflow as tf
import numpy as np


class TerminateOnNaNOrInf(tf.keras.callbacks.Callback):
    def __init__(self, monitor="loss"):
        super().__init__()
        self.monitor = monitor

    def on_batch_end(self, batch, logs=None):
        if math.isnan(logs[self.monitor]) or math.isinf(logs[self.monitor]):
            self.model.stop_training = True
            print("Batch %d: Invalid loss, terminating training" % (batch))


class StepLearningRateOnEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, d_optimizer, factor=0.1, patience=5):
        super().__init__()
        self.d_optimizer = d_optimizer
        self.factor = factor
        self.patience = patience
        self.wait = 0

    def on_train_end(self, logs=None):
        # Early stopping changes the model.stop_training
        # This does not activate the callback when the model stops training due to epoch
        if self.model.stop_training:
            d_lr = self.d_optimizer.lr.numpy()
            new_d_lr = d_lr * self.factor
            tf.keras.backend.set_value(self.d_optimizer.lr, new_d_lr)
            print(f"Learning rate stepped down. Discriminator LR: {new_d_lr}")
            self.wait += 1

    @property
    def stop(self):
        return self.wait >= self.patience


def generate_for_callback(
    model: tf.keras.Model,
    test_input: np.ndarray,
    tar: np.ndarray,
    writer: tf.summary.SummaryWriter,
):
    """
    Generates the function for the callback to use.

    Parameters
    ----------
    model :
        _description_
    test_input : _type_
        _description_
    tar : _type_
        _description_
    writer : _type_
        _description_
    """

    def generate_images_tensorboard(epoch, logs=None):
        if epoch % 100 != 0:
            # Experiment in context manager
            with writer.as_default():
                prediction = model(test_input, training=True)
                display_list = [test_input[0], tar[0], prediction[0]]
                display_list = [
                    tf.image.convert_image_dtype(x, tf.float64) for x in display_list
                ]
                display_list = [x * 0.5 + 0.5 for x in display_list]
                titles = ["Input Image", "Ground Truth", "Predicted Image"]
                tf.summary.image(
                    " ".join(titles), display_list, step=epoch, max_outputs=6
                )
                writer.flush()
        if epoch % 1000 == 0 and epoch != 0:
            tf.keras.backend.clear_session()

    return lambda epoch, logs: generate_images_tensorboard(epoch, logs)


class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        generator: tf.keras.Model,
        log_dir: str,
        test_input: np.ndarray,
        test_target: np.ndarray,
    ):
        super(ImageCallback, self).__init__()
        self.generator = generator
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(self.log_dir)
        self.test_input = test_input
        self.test_target = test_target

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            with self.file_writer.as_default():
                # Generate images using the generator model
                prediction = self.generator.predict(self.test_input)

                # Create a grid of 4x4 images
                display_list = [self.test_input[0], self.test_target[0], prediction[0]]
                display_list = [
                    tf.image.convert_image_dtype(x, tf.float64) for x in display_list
                ]
                display_list = [x * 0.5 + 0.5 for x in display_list]
                titles = ["Input Image", "Ground Truth", "Predicted Image"]

                # Log the generated image to TensorBoard
                with self.file_writer.as_default():
                    tf.summary.image(
                        " ".join(titles), display_list, step=epoch, max_outputs=6
                    )
                self.file_writer.flush()

                # Flush the summary writer
                self.file_writer.flush()
                self.image_count += 1

        # Clear the session to free up memory
        if epoch % 1000 == 0 and epoch != 0:
            tf.keras.backend.clear_session()


def VGG19Generator(num_classes=3, trainable=False):
    # Load VGG19 model pretrained on ImageNet without the top layers
    vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=(None, None, 3))

    # Set VGG19 layers to not trainable
    if not trainable:
        for layer in vgg19.layers:
            layer.trainable = False
    else:
        for layer in vgg19.layers:
            layer.trainable = True

    # Get the output of each block of VGG19
    block1 = vgg19.get_layer("block1_conv2").output  # (256 x 256 x 64)
    block2 = vgg19.get_layer("block2_conv2").output  # (128 x 128 x 128)
    block3 = vgg19.get_layer("block3_conv4").output  # (64 x 64 x 256)
    block4 = vgg19.get_layer("block4_conv4").output  # (32 x 32 x 512)
    # block5 = vgg19.get_layer("block5_conv4").output  # (16 x 16 x 512)

    # Upsampling layers

    # up_conv5 = _upsample(512, 3)(block5)  # (32 x 32 x 512)
    # up_concat5 = tf.keras.layers.concatenate([up_conv5, block4])  # (32 x 32 x 1024)
    up_conv1 = _upsample(256, 3)(block4)  # (64 x 64 x 256)
    up_concat1 = tf.keras.layers.concatenate([up_conv1, block3])  # (64 x 64 x 512)
    up_conv2 = _upsample(128, 3)(up_concat1)  # (128 x 128 x 128)
    up_concat2 = tf.keras.layers.concatenate([up_conv2, block2])  # (128 x 128 x 256)
    up_conv3 = _upsample(64, 3)(up_concat2)  # (256 x 256 x 64)
    up_concat3 = tf.keras.layers.concatenate([up_conv3, block1])  # (256 x 256 x 128)

    # Output layer
    output_layer = tf.keras.layers.Conv2D(num_classes, 1, activation="sigmoid")(
        up_concat3
    )  # (256 x 256 x num_classes)

    # Define the model
    model = tf.keras.models.Model(inputs=vgg19.input, outputs=output_layer)

    return model


def VGG19Discriminator(trainable=False):
    inp = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")

    # VGG19 architecture
    vgg = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=[256, 256, 3]
    )
    if not trainable:
        for layer in vgg.layers:
            layer.trainable = False
    else:
        for layer in vgg.layers:
            layer.trainable = True

    inp_ = vgg(inp)
    tar_ = vgg(tar)

    x = tf.keras.layers.concatenate([inp_, tar_])  # (batch_size, 256, 256, channels*2)
    # Additional trainable layers
    x = tf.keras.layers.Flatten()(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def _downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def _upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[None, None, 3])

    down_stack = [
        _downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        _downsample(128, 4),  # (batch_size, 64, 64, 128)
        _downsample(256, 4),  # (batch_size, 32, 32, 256)
        _downsample(512, 4),  # (batch_size, 16, 16, 512)
        _downsample(512, 4),  # (batch_size, 8, 8, 512)
        _downsample(512, 4),  # (batch_size, 4, 4, 512)
        _downsample(512, 4),  # (batch_size, 2, 2, 512)
        _downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        _upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        _upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        _upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        _upsample(512, 4),  # (batch_size, 16, 16, 1024)
        _upsample(256, 4),  # (batch_size, 32, 32, 512)
        _upsample(128, 4),  # (batch_size, 64, 64, 256)
        _upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        kernel_initializer=initializer,
        padding="same",
        activation="sigmoid",
    )  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = _downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = _downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = _downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.gen_gan_loss_tracker = tf.keras.metrics.Mean(name="gen_gan_loss")
        self.gen_l1_loss_tracker = tf.keras.metrics.Mean(name="gen_l1_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.vgg = self.builds_vgg()
        self.vgg.trainable = False

    def builds_vgg(self):
        """
        Build the VGG19 model for perceptual loss

        Returns
        -------
        tf.keras.Model
            VGG19 model
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        outputs = [
            vgg.get_layer(name).output for name in ["block1_conv2", "block2_conv2", "block3_conv4"]
        ]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def compile(
        self,
        g_optimizer: tf.keras.optimizers,
        d_optimizer: tf.keras.optimizers,
        loss_fn: tf.keras.losses,
    ):
        """
        Compile the model

        Parameters
        ----------
        g_optimizer : tf.keras.optimizers
            Generator optimizer
        d_optimizer : tf.keras.optimizers
            Discriminator optimizer
        loss_fn : tf.keras.losses
            Loss function
        """
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, input):
        """
        train step used in the fit method

        Parameters
        ----------
        input : tuple
            This is highly dependent on the data you are using.
            In this case, it must be a tuple of (input_image, target).

        Returns
        -------
        dict
            Dictionary containing the losses for the generator and discriminator.
        """
        input_image, target = input
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True
            )

            gen_perceptual_loss = self.perceptual_loss(
                target=target, generated=gen_output
            )

            g_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target
            )
            g_loss = g_loss + gen_perceptual_loss / 2
            d_loss = self.discriminator_loss(disc_real_output,disc_generated_output)

        generator_gradients = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.gen_gan_loss_tracker.update_state(gen_gan_loss)
        self.gen_l1_loss_tracker.update_state(gen_l1_loss)
        self.perceptual_loss_tracker.update_state(gen_perceptual_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "gen_gan_loss": self.gen_gan_loss_tracker.result(),
            "gen_l1_loss": self.gen_l1_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
        }

    def generator_loss(self, disc_generated_output, gen_output, target, LAMBDA=100):
        """
        L1 loss and GAN loss are two different types of loss functions commonly used
        in Generative Adversarial Networks (GANs).

        L1 loss encourages the generator to produce images that are similar
        to the target images in terms of their pixel values, while GAN loss encourages
        the generator to produce images that can fool the discriminator into thinking
        that they are real.

        A higher lambda value means that the generator will generate images that are
        more similar to the target images.

        Parameters
        ----------
        disc_generated_output :
            Discriminator output for generated images
        gen_output :
            Generated image
        target :
            Target image

        Returns
        -------
        float
            Loss value
        """
        gan_loss = self.loss_fn(
            tf.ones_like(disc_generated_output), disc_generated_output
        )

        # Mean absolute error
        l1_loss = tf.reduce_mean(
            tf.abs(tf.cast(target, tf.float32) - tf.cast(gen_output, tf.float32))
        )

        total_gen_loss = tf.cast(gan_loss, tf.float32) + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Discriminator loss function

        Parameters
        ----------
        disc_real_output : image
            Real image
        disc_generated_output : image
            Generated image

        Returns
        -------
        float
            Loss value
        """
        real_loss = self.loss_fn(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_fn(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def perceptual_loss(self, target, generated):
        """
        Perceptual loss function: Compares the features generated from the target and
        generated images by the VGG19 network. The perceptual loss is the mean squared
        error between the two sets of features.

        Parameters
        ----------
        target : image
            Target image
        generated : image
            Generated image

        Returns
        -------
        float
            Loss value
        """
        # Get the VGG-19 features for the target and generated images
        target_features = self.vgg(target)
        generated_features = self.vgg(generated)

        # Compute the mean squared error between the target and generated features
        loss = 0
        for t, g in zip(target_features, generated_features):
            loss += tf.reduce_mean(tf.square(t - g))
        loss /= len(target_features)

        return tf.cast(loss, tf.float32)

    def call(self, inputs, training=None, mask=None):
        # Not used
        """
        Call method used in the fit method

        Parameters
        ----------
        inputs : tuple
            This is highly dependent on the data you are using.
            In this case, it must be a tuple of (input_image, target).
        training : bool, optional
            Whether the model is training or not, by default None
        mask : None, optional
            by default None

        Returns
        -------
        dict
            Dictionary containing the losses for the generator and discriminator.
        """

        return self.generator(inputs, training=training)
