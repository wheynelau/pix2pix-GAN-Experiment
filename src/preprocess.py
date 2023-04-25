import os
from tqdm import tqdm
import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf
# Run this script in the root directory of the project

# Assumes the data is in data/train and data/test
# Assumes the data is in the format of image.png and mask.png
# _load function needs masks in the data/mask folder

def _load(image_file, mask:bool = True):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)
    # Read and decode the mask file to a uint8 tensor
    # If there is no mask folder, the
    image_file = tf.strings.regex_replace(image_file, "train", "mask")
    image_file = tf.strings.regex_replace(image_file, "test", "mask")
    image_mask = None
    if mask:
        if "mask" not in image_file.numpy().decode():
            raise ValueError("No mask folder found")

        image_mask = tf.io.read_file(image_file)
        image_mask = tf.io.decode_png(image_mask)

        # Convert the mask to a binary mask
        # Using a mask of 0 and 255 instead of 0 and 1
        # This makes all the background white
        image_mask = tf.repeat(image_mask, 3, axis=2)
        image_mask = tf.where(image_mask > 0, 0, 255)
    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    # - Add the masks to the image to remove background
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    if mask:
        image_mask = tf.cast(image_mask, tf.float32)
        real_image = tf.add(real_image, image_mask)
    real_image = tf.clip_by_value(real_image, 0, 255)

    return input_image, real_image


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def preprocess(args: DictConfig):
    """
    Runs the preprocessing script
    Splits the image into image and target
    """
    if args.preprocess.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Check args
    if not os.path.exists(args.preprocess.data_path):
        raise ValueError("Data path does not exist")

    orig_train_images_path = [
        os.path.join(args.preprocess.data_path, "train", x)
        for x in os.listdir(os.path.join(args.preprocess.data_path, "train"))
        if "-" not in x
    ]
    orig_test_images_path = [
        os.path.join(args.preprocess.data_path, "test", x)
        for x in os.listdir(os.path.join(args.preprocess.data_path, "test"))
    ]

    # create a new directory to store the preprocessed images

    train_image_path = os.path.join(args.preprocess.preprocess_path, "train", "image/")
    train_mask_path = os.path.join(args.preprocess.preprocess_path, "train", "target/")
    test_image_path = os.path.join(args.preprocess.preprocess_path, "test", "image/")
    test_mask_path = os.path.join(args.preprocess.preprocess_path, "test", "target/")

    # create the directories using os.makedirs
    for x in [train_image_path, train_mask_path, test_image_path, test_mask_path]:
        os.makedirs(x, exist_ok=True)

    # preprocess the train images and save them into the new directory

    for i, image_path in enumerate(tqdm(orig_train_images_path)):
        # save into a new directory
        x, y = _load(image_path, args.preprocess.mask)
        tf.keras.utils.save_img(train_image_path + str(i) + ".png", x)
        tf.keras.utils.save_img(train_mask_path + str(i) + ".png", y)
    # preprocess the test images and save them into the new directory
    for i, image_path in enumerate(tqdm(orig_test_images_path)):
        # save into a new directory
        x, y = _load(image_path, args.preprocess.mask)
        tf.keras.utils.save_img(test_image_path + str(i) + ".png", x)
        tf.keras.utils.save_img(test_mask_path + str(i) + ".png", y)


if __name__ == "__main__":
    preprocess()
