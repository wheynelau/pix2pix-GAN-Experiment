from ..src.models import VGG19Generator
import os
import numpy as np
import tensorflow as tf # replace with your generator class name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow logging

# Check code style using flake8
def test_flake8():
    assert os.system('flake8 --ignore=E501 src/model.py') == 0

# Test if the generator can load and predict an image without errors
def test_generator():
    # Generate random image dimensions
    height = tf.random.uniform([], maxval=2000, dtype=tf.int32)
    width = tf.random.uniform([], maxval=2000, dtype=tf.int32)
    
    # Create a random noise vector for input to the generator
    pixels = tf.random.uniform([height, width, 3], minval=0, maxval=256, dtype=tf.int32)
    
    # Instantiate generator
    generator = VGG19Generator()
    
    # Load the generator weights from file
    # Base model
    generator.load_weights('..models/generator.h5')
    
    # Preprocess the input
    pixels /= 255.0

    # Generate an image from the input noise vector
    output = generator.predict(pixels)
    output = output * 255
    output = np.squeeze(output, axis=0)
    output_image = output.astype(np.uint8)
    
    # Ensure the generated image has the correct shape
    assert output_image.shape == (1, height, width, 3)
    
    # Ensure the generated image has no NaN or inf values
    assert not np.isnan(output_image).any()
    assert not np.isinf(output_image).any()
    
    print('Generator test passed.')
