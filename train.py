import os
import gc
import datetime
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy("mixed_float16")
from tqdm import tqdm
from src.models import *
from src.utils import TFUtils
import argparse
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose",
    "-vb",
    type=int,
    default=0,
    help="Verbose output during training",
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=100,
    help="Number of epochs to train the model for",
)
parser.add_argument(
    "--steps",
    "-s",
    type=int,
    default=10,
    help="Number of steps per epoch to train the model for",
)
parser.add_argument(
    "--runs",
    "-r",
    type=int,
    default=100,
    help="Number of runs to train the model for",
)
parser.add_argument(
    "--batch_size",
    "-b",
    type=int,
    default=8,
    help="Batch size to use for training",
)
parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=0.0002,
    help="Learning rate to use for training",
)
parser.add_argument(
    "--down_factor",
    "-df",
    type=float,
    default=0.1,
    help="Downsampling factor to use for the discriminator",
)
parser.add_argument(
    "--width",
    "-wt",
    type=int,
    default=256,
    help="Width of the images to use for training",
)
parser.add_argument(
    "--height",
    "-ht",
    type=int,
    default=256,
    help="Height of the images to use for training",
)
parser.add_argument(
    "--load",
    "-l",
    action="store_true",
    help="Load the latest checkpoint and continue training",
)
parser.add_argument(
    "--vgg",
    "-vgg",
    action="store_true",
    help="Use VGG models",
)

args = parser.parse_args()

# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = args.batch_size
# Each image is 256x256 in size
IMG_WIDTH = args.width
IMG_HEIGHT = args.height
STEPS = args.steps
NUM_RUNS = args.runs
EPOCHS = args.epochs
DOWN_FACTOR = args.down_factor
utils = TFUtils(args.vgg)

# Create the generator and discriminator
train_generator, validation_generator = utils.create_datagenerators(
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
)
generator = Generator()
discriminator = Discriminator()

# Create the optimizers
generator_optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.6)
discriminator_optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.6)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gan = GAN(generator = generator, discriminator = discriminator)
gan.compile(g_optimizer = generator_optimizer, d_optimizer = discriminator_optimizer,loss_fn = loss_object)

os.makedirs("logs", exist_ok=True)
os.makedirs("training_checkpoints", exist_ok=True)
# Create the checkpoint directory
if args.load:
    try:
        # load the model from the latest checkpoint
        time_now = os.listdir('training_checkpoints')[-1]
        latest = tf.train.latest_checkpoint(os.path.join('training_checkpoints', time_now))
        print("Loading model from: ", latest)
        gan.load_weights(latest)
        # load the logging folder
        log_dir = os.path.join("logs", max(os.listdir('logs')))
        print("Loading logs from: ", log_dir)
    except IndexError:
        print("No checkpoints found. Training from scratch")
        time_now = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
else:
    time_now = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    log_dir = os.path.join("logs", time_now)

ckpt_dir = os.path.join("./training_checkpoints",str(datetime.datetime.now().strftime("%Y%m%d-%H%M")), "ckpt")
print("New model will be saved to: ", ckpt_dir)
checkpoint_prefix = ckpt_dir

def main():

    ### CHECKPOINTS ###
    example_input, example_target = next(iter(validation_generator))

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='g_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True
)
        
    tf_summary = tf.summary.create_file_writer(os.path.join(log_dir,'images'))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        write_graph=False,)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='gen_gan_loss', restore_best_weights=True,
                                                    mode='min', baseline=100, verbose=1, patience=20)

    lr_scheduler_d = StepLearningRateOnEarlyStopping(discriminator_optimizer, factor= DOWN_FACTOR)

    #### TRAINING LOOP ####
    start = time.time()
    for i in range(NUM_RUNS):
        if i != 0:
            print(f"Time taken for Run: {(time.time() - start)/60:2f} mins")
            start = time.time()
        print(f"Run: {i} / {NUM_RUNS}")
        gan.fit(
            train_generator,
            epochs=EPOCHS,
            steps_per_epoch=STEPS,
            use_multiprocessing=True,
            verbose = args.verbose,
            callbacks=[checkpoint_callback, 
            tensorboard_callback, 
            es_callback,
            lr_scheduler_d,
            #terminate
            ],
        )
        utils.generate_images_tensorboard(gan, example_input, example_target, tf_summary, i)
        gc.collect()
        tf.keras.backend.clear_session()

    # Save the model
    gen = gan.generator
    gen.save(os.path.join("models", time_now, "generator.h5"))



if __name__ == "__main__":
    main()
