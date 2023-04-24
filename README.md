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
- Create a training pipeline for GAN models

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

```
_Note: VGG was not trained_

# Inferencing

At the end of train.py, I've added a statement to save the generator of the GAN model.
This is the generator that will be used for inferencing. 

Run the below command to infer on a folder containing images:

_Note: There is no exception handling for non-image files, please input only image files_

```bash
$ python infer.py --help
usage: infer.py [-h] [--cpu] [--concat] image_path output_path

```
# Results

1. The generator was able to generate rather realistic images however they were incomplete. They looked similar to incomplete paintings.
2. Despite not having colour information, the generator was able to generate images with similar colours to the original images.
  - This could mean that the generator was able to learn the colour information from the features
3. The model could not learn the features of the original images even after adding perception loss
   - More experimentation needs to be carried over -> moved into TODO

Here are some of the samples, as mentioned, more experimentation needs to be carried out to improve the results.

<img src ="images/9.png" title='Initial run with the original pix2pix architecture' alt width="50%">

> Initial run with the original pix2pix architecture

<img src ="images/9_percep.png" title='Run with VGG generator, default discriminator and perceptual loss' width="50%">

> Run with VGG generator, default discriminator and perceptual loss

Further details on the perceptual loss:

The VGG features are from the first 3 conv outputs before the max pool layers.

Thereafter, the mean squared error is calculated between the features of the original image and the generated image.

The perceptual loss was given a weight of 0.5 and the lambda value for the l1 loss function was set to 100.

# Problems
1. VGG implementation of generator was successful
   - However, the performance was not great

# Learning points
1. GANs are hard to train
2. It is difficult to tune the learning rates for the generator and discriminator
3. Successfully implemented custom callbacks for Tensorboard and checkpointing
4. Created a callback to adjust the learning rate of the generator and discriminator
# Future enhancements / TODO

1. Implement a mix of pretrained models for the discriminator and generator
   - Using VGG for generator and using a custom discriminator vice versa
   - UPDATE: Implemented VGG for generator and custom discriminator
2. Experiment with different lambda values for the loss function
   - Hypothesis: Generate images that are realistic but not similar to the original images
3. Implement hydra for configuration management
4. Increase customisability of the training pipeline
   - Use custom file names : currently the file names are hardcoded, such as preprocessed/train/image and preprocessed/train/mask
   - Add more options for the loss function
   - UPDATE: Implemented perceptual loss

# Contributing

Feedback and contributions are welcome. As this is a learning project, I may not be active in maintaining this repository and accepting pull requests. 

# Sources 

[pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix)
[APDrawingGAN](https://github.com/yiranran/APDrawingGAN/) # Found the datasets from this repository
