# UNET GAN

# Table Of Contents
- [UNET GAN](#unet-gan)
- [Table Of Contents](#table-of-contents)
- [Motivation and goals](#motivation-and-goals)
- [Pre-requisites](#pre-requisites)
- [Getting started](#getting-started)
- [Training](#training)
- [Inferencing](#inferencing)
- [Results](#results)
- [Problems](#problems)
- [Learning points](#learning-points)
- [Future enhancements / TODO](#future-enhancements--todo)
- [Contributing](#contributing)
- [Sources](#sources)

# Motivation and goals

- Explore the use cases of GAN
- Experiment with cloud computing resources
- Implement a GAN model with tensorflow
- Use pretrained models as a discriminator and generator

# Pre-requisites

Tensorflow was built from source using the following configuration:

```
python=3.10
tensorflow=2.9.3
cudatoolkit=11.2
cudnn=8.1
```

A conda environment file will be provided in the root directory of this repository. It was only tested on a windows machine.

# Getting started
If you want to use this project, follow these steps:

1. Clone this repository

```bash
git clone https://github.com/wheynelau/VGG19-gan-experiment.git
```
2. Install the requirements, before this you would need conda installed on your machine. You can install conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

```bash
conda env create -f environment.yml
```
3. Setup the folders and files

If your image is in the format of two images combined together, you can use the 'preprocess.py' file to split them into two images. 

Here is an example of the image:

<img src ="images/sample.png" title='Sample' width="50%">

Your directory should look like this:

```bash
python src/preprocess.py
├───data
│   ├───mask
│   ├───test
│   └───train
```

Running the 'preprocess.py' file will create a new directory called 'preprocessed' and split the images into two images. This is how it would appear:

```bash
├───preprocessed
│   ├───test
│   │   ├───image
│   │   └───mask
│   └───train
│       ├───image
│       └───mask
```
# Training

```bash
$ python train.py --help
usage: train.py [-h] [--verbose VERBOSE] [--epochs EPOCHS] [--steps STEPS] [--runs RUNS]
                [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--down_factor DOWN_FACTOR]        
                [--width WIDTH] [--height HEIGHT] [--load] [--vgg]
optional arguments:
  -h, --help            show this help message and exit
  --verbose VERBOSE, -vb VERBOSE
                        Verbose output during training
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs to train the model for
  --steps STEPS, -s STEPS
                        Number of steps per epoch to train the model for
  --runs RUNS, -r RUNS  Number of runs to train the model for
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size to use for training
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate to use for training
  --down_factor DOWN_FACTOR, -df DOWN_FACTOR
                        Downsampling factor to use for the discriminator
  --width WIDTH, -wt WIDTH
                        Width of the images to use for training
  --height HEIGHT, -ht HEIGHT
                        Height of the images to use for training
  --load, -l            Load the latest checkpoint and continue training
  --vgg, -vgg           Use VGG models
```
Note: VGG was not trained

# Inferencing

At the end of train.py, I've added a statement to save the generator of the GAN model.
This is the generator that will be used for inferencing.

```bash
$ python infer.py --help
usage: infer.py [-h] [--cpu] [--concat] image_path output_path

Inference on an image using a saved TensorFlow model.

positional arguments:
  image_path   path to the input folder
  output_path  path to the model folder

options:
  -h, --help   show this help message and exit
  --cpu        use CPU instead of GPU
  --concat     concatenate the input and output images
```
# Results

Tensorboard results can be found [here](https://tensorboard.dev/experiment/gk65vsEQQDuOCeGJv2Yt1A/)

1. The generator was able to generate rather realistic images however they were incomplete. They looked similar to incomplete paintings.
2. Despite not having colour information, the generator was able to generate images with similar colours to the original images.
  - This could mean that the generator was able to learn the colour information from the features

# Problems
1. VGG implementation was not successful, leading to inf loss on the generator
   - Suspected that the discriminator was overpowering the generator

# Learning points
1. GANs are hard to train
2. It is difficult to tune the learning rates for the generator and discriminator
3. Successfully implemented custom callbacks for Tensorboard and checkpointing
4. Created a callback to adjust the learning rate of the generator and discriminator
# Future enhancements / TODO

1. Implement a mix of pretrained models for the discriminator and generator
   - Using VGG for generator and using a custom discriminator vice versa
2. Experiment with different lambda values for the loss function
   - Hypothesis: Generate images that are realistic but not similar to the original images
3. Implement hydra for configuration management
4. Fine tune a pretrained generator and discriminator

# Contributing

Any kind of enhancement or contribution is welcomed.

# Sources 

[pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix)

