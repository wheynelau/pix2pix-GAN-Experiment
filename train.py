import os
import gc
import argparse
import json
import datetime
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")
from tqdm import tqdm
from src.models import *
from src.utils import TFUtils
import hydra
from omegaconf import DictConfig, OmegaConf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(args: DictConfig):
    # Assign the arguments
    BATCH_SIZE = args.train.batch_size
    IMG_WIDTH = args.train.width
    IMG_HEIGHT = args.train.height
    STEPS = args.train.steps
    NUM_RUNS = args.train.runs
    EPOCHS = args.train.epochs
    DOWN_FACTOR = args.train.down_factor

    # Create the utils object

    utils = TFUtils(
        args.train.vgg,
        args.preprocess.preprocess_path,
        args.train.noise,
        args.train.noise_amount,
    )

    # Interrupt mode: if the training is interrupted, the learning rate will be loaded from the previous learning rate
    # and the training will continue from the last checkpoint
    if args.train.interrupt_mode and args.train.lr_optimizer == 'discriminator':
        try:
            with open('myfile.txt', "r") as f:
                first_line = f.readline()
            discriminator_learning = float(first_line)
        except FileNotFoundError:
            print("No file found, starting from config file")
            discriminator_learning = args.train.learning_rate * args.train.discriminator_factor
        finally:
            generator_learning = args.train.learning_rate
    elif args.train.interrupt_mode and args.train.lr_optimizer == 'generator':
        try:
            with open('myfile.txt', "r") as f:
                first_line = f.readline()
            generator_learning = float(first_line)
        except FileNotFoundError:
            print("No file found, starting from config file")
            generator_learning = args.train.learning_rate
        finally:
            discriminator_learning = args.train.learning_rate * args.train.discriminator_factor
    else:
        generator_learning = args.train.learning_rate
        discriminator_learning = args.train.learning_rate * args.train.discriminator_factor

    # Create the dataset
    train_generator, validation_generator = utils.create_datagenerators(
        IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
    )

    # Create the generator and discriminator
    if args.train.vgg:
        generator = VGG19Generator()
        discriminator = VGG19Discriminator()
    else:
        generator = Generator()
        discriminator = Discriminator()

    # Create the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(
        generator_learning, beta_1=args.train.gen_beta1
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        discriminator_learning,
        beta_1=args.train.disc_beta1,
    )
    
    # Create the GAN
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan = GAN(
        generator=generator,
        discriminator=discriminator,
        l1lambda=args.train.l1lambda,
        perceptual_weight= args.train.perceptual_weight,
    )
    # Compile the GAN
    gan.compile(
        g_optimizer=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        loss_fn=loss_object,
    )
    # Creating the logs and checkpoints directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("training_checkpoints", exist_ok=True)

    # Loading the checkpoints and logs
    if args.train.load:
        try:
            # load the model from the latest checkpoint
            time_now = os.listdir("training_checkpoints")[-1]
            latest = tf.train.latest_checkpoint(
                os.path.join("training_checkpoints", time_now)
            )
            print("Loading model from: ", latest)
            gan.load_weights(latest)
            # load the logging folder
            log_dir = os.path.join("logs", max(os.listdir("logs")))
            print("Loading logs from: ", log_dir)
            os.makedirs(log_dir, exist_ok=True)
        except IndexError:
            print("No checkpoints found. Training from scratch")
            time_now = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
            log_dir = os.path.join("logs", time_now)
    else:
        time_now = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        log_dir = os.path.join("logs", time_now)

    # A new checkpoint has to be created for each run

    ckpt_dir = os.path.join(
        "./training_checkpoints",
        str(datetime.datetime.now().strftime("%Y%m%d-%H%M")),
        "ckpt",
    )
    print("New model will be saved to: ", ckpt_dir)
    checkpoint_prefix = ckpt_dir

    # Set the vgg layers to trainable or not

    if args.train.vgg_trainable:
        for layer in gan.generator.layers:
            layer.trainable = True

    ### CHECKPOINTS ###
    example_input, example_target = next(iter(validation_generator))

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        monitor="g_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Uncomment this to save the images to tensorboard
    # tf_summary = tf.summary.create_file_writer(os.path.join(log_dir, "images"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="d_loss",
        restore_best_weights=False,
        mode="min",
        baseline=100,
        verbose=1,
        patience=50,
    )

    lr_scheduler_d = StepLearningRateOnEarlyStopping(
        discriminator_optimizer if args.train.lr_optimizer.lower() == "discriminator" else generator_optimizer,
        factor=DOWN_FACTOR, patience=30, discriminator_save_path='optimiser.txt'
    )

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
            verbose=args.train.verbose,
            callbacks=[
                checkpoint_callback,
                tensorboard_callback,
                es_callback,
                lr_scheduler_d,
                # terminate
            ],
        )
        # Uncomment this to save the images to tensorboard
        # Generate images on tensorboard and clear the session
        # utils.generate_images_tensorboard(
        #    gan, example_input, example_target, tf_summary, i
        #)
        gc.collect()
        tf.keras.backend.clear_session()
        if lr_scheduler_d.stop:
            print("Early stopping after 30 runs of downfactor")
            break

        # Save the model
        gen = gan.generator
        gen.save(os.path.join("models", time_now, "generator.h5"))


if __name__ == "__main__":
    train()
