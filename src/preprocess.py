import os
from tqdm import tqdm
import tensorflow as tf
import argparse
# Run this script in the root directory of the project

# Assumes the data is in data/train and data/test
# Assumes the data is in the format of image.png and mask.png
# _load function needs masks in the data/mask folder

def _load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)
    # Read and decode the mask file to a uint8 tensor
    # If there is no mask folder, the 
    image_file = tf.strings.regex_replace(image_file, "train", "mask")
    image_file = tf.strings.regex_replace(image_file, "test", "mask")
    
    if 'mask' not in image_file.numpy().decode():
        raise ValueError("No mask folder found")

    mask = tf.io.read_file(image_file)
    mask = tf.io.decode_png(mask)

    # Convert the mask to a binary mask
    # Using a mask of 0 and 255 instead of 0 and 1
    # This makes all the background white
    mask = tf.repeat(mask, 3, axis=2)
    mask = tf.where(mask > 0, 0, 255)
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
    mask = tf.cast(mask, tf.float32)
    real_image = tf.add(real_image, mask)
    real_image = tf.clip_by_value(real_image, 0, 255)

    return input_image, real_image


def main(preprocess_path:str, data_path:str):
    """
    Runs the preprocessing script
    Splits the image into 
    """

    train_images_path = [os.path.join(data_path, 'train', x) for x in os.listdir(os.path.join(data_path, 'train')) if "-" not in x]
    test_images_path = [os.path.join(data_path, 'test', x) for x in os.listdir(os.path.join(data_path, 'test'))]

    # create a new directory to store the preprocessed images

    train_image_path = os.path.join(preprocess_path, "train", "image")
    train_mask_path = os.path.join(preprocess_path, "train", "target")
    test_image_path = os.path.join(preprocess_path, "test", "image")
    test_mask_path = os.path.join(preprocess_path, "test", "target")

    # create the directories using os.makedirs
    for x in [train_image_path, train_mask_path, test_image_path, test_mask_path]:
        os.makedirs(x, exist_ok=True)

    # preprocess the train images and save them into the new directory

    for i, image_path in enumerate(tqdm(train_images_path)):
        # save into a new directory
        x, y = _load(image_path)
        tf.keras.utils.save_img(train_image_path + str(i) + ".png", x)
        tf.keras.utils.save_img(train_mask_path + str(i) + ".png", y)
    # preprocess the test images and save them into the new directory
    for i, image_path in enumerate(tqdm(test_images_path)):
        # save into a new directory
        x, y = _load(image_path)
        tf.keras.utils.save_img(test_image_path + str(i) + ".png", x)
        tf.keras.utils.save_img(test_mask_path + str(i) + ".png", y)

if __name__ == "__main__":

    args= argparse.ArgumentParser()
    args.add_argument("data_path", type=str, default="data")
    args.add_argument("preprocess_path", type=str, default="preprocess")
    args = args.parse_args()

    # check args

    if not os.path.exists(args.data_path):
        raise ValueError("Data path does not exist")

    if not os.path.exists(args.preprocess_path):
        os.makedirs(args.preprocess_path)

    main(args.preprocess_path, args.data_path)