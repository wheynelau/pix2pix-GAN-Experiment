import tensorflow as tf
import matplotlib.pyplot as plt
import os


class TFUtils:

    def __init__(self, vgg:bool, preprocess_path:str, noise_flag:bool, noise_amount:float = 1):
        self.vgg = vgg
        self.preprocessor_path = preprocess_path
        self.noise = noise_flag
        if self.noise and (noise_amount <= 0 or noise_amount > 1):
            raise ValueError("Noise amount must be 0<noise_amount<=1")
        self.noise_amount = noise_amount
    def preprocessor(self,img):
            return img /255
    
    def preprocessor_train(self,img):
        img /= 255
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0, dtype=tf.float32) * self.noise_amount
        noisy_image = tf.clip_by_value(img + noise, 0.0, 1.0)
        return noisy_image

    def create_datagenerators(self,height, width, bs):
        seed = 1
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="reflect",
            preprocessing_function=self.preprocessor_train if self.noise else self.preprocessor,
        )
        train_generator_image = train_datagen.flow_from_directory(
            os.path.join(self.preprocessor_path,"train"),
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            color_mode="rgb",
            classes=["image"],
            seed=seed,
        )
        train_generator_mask = train_datagen.flow_from_directory(
            os.path.join(self.preprocessor_path,"train"),
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            color_mode="rgb",
            classes=["target"],
            seed=seed,
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function =self.preprocessor
            )
        validation_generator_images = test_datagen.flow_from_directory(
            os.path.join(self.preprocessor_path,"test"),
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            classes=["image"],
            color_mode="rgb",
            seed=seed,
        )
        validation_generator_masks = test_datagen.flow_from_directory(
            os.path.join(self.preprocessor_path,"test"),
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            classes=["target"],
            color_mode="rgb",
            seed=seed,
        )

        # Zip the image and mask generators
        train_generator = tf.data.Dataset.zip((tf.data.Dataset.from_generator(lambda: train_generator_image, output_types=tf.float32, output_shapes=[None, 256, 256, 3]),
                                        tf.data.Dataset.from_generator(lambda: train_generator_mask, output_types=tf.float32, output_shapes=[None, 256, 256, 3])))
        validation_generator = zip(validation_generator_images, validation_generator_masks)

        return train_generator.cache(), validation_generator

    @staticmethod
    def generate_images(model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        display_list = [tf.image.convert_image_dtype(x, tf.float64) for x in display_list]
        title = ["Input Image", "Ground Truth", "Predicted Image"]

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")
        plt.show()

    def generate_images_tensorboard(self,model, test_input, tar,writer,step):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        display_list = [tf.image.convert_image_dtype(x, tf.float64) for x in display_list]
        display_list = [self.reversev2(x) for x in display_list]
        titles = ["Input Image", "Ground Truth", "Predicted Image"]
        
        with writer.as_default():
            tf.summary.image(' '.join(titles), display_list, step=step, max_outputs=6)
        writer.flush()


    @staticmethod
    def reversev2(input_image):

        input_image *= 255
        input_image = tf.clip_by_value(input_image, 0, 255)

        return tf.cast(input_image, tf.uint8)