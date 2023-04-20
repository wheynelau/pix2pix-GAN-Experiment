import tensorflow as tf
import matplotlib.pyplot as plt


class TFUtils:

    def __init__(self, vgg:bool):
        self.vgg = vgg

    def preprocessor(self,img):
            if self.vgg:
                img = tf.keras.applications.vgg19.preprocess_input(img)
            else:
                img = img/127.5 - 1
            return img

    def create_datagenerators(self,height, width, bs):
        seed = 1
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="reflect",
            preprocessing_function=self.preprocessor,
        )
        train_generator_image = train_datagen.flow_from_directory(
            "preprocessed/train",
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            color_mode="rgb",
            classes=["image"],
            seed=seed,
        )
        train_generator_mask = train_datagen.flow_from_directory(
            "preprocessed/train",
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            color_mode="rgb",
            classes=["mask"],
            seed=seed,
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function =self.preprocessor)
        validation_generator_images = test_datagen.flow_from_directory(
            "preprocessed/test",
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            classes=["image"],
            color_mode="rgb",
            seed=seed,
        )
        validation_generator_masks = test_datagen.flow_from_directory(
            "preprocessed/test",
            target_size=(height, width),
            batch_size=bs,
            class_mode=None,
            classes=["mask"],
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

    @staticmethod
    def generate_images_tensorboard(model, test_input, tar,writer,step):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        display_list = [tf.image.convert_image_dtype(x, tf.float64) for x in display_list]
        display_list = [x* 0.5 + 0.5 for x in display_list]
        titles = ["Input Image", "Ground Truth", "Predicted Image"]
        
        with writer.as_default():
            tf.summary.image(' '.join(titles), display_list, step=step, max_outputs=6)
        writer.flush()