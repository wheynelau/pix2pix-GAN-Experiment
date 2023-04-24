import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import VGG19Generator
import numpy as np
import tensorflow as tf # replace with your generator class name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow logging

# Test if the generator can load and predict an image without errors
def test_generator():
    
    # Create a random noise vector for input to the generator
    pixels = tf.random.uniform([256, 256, 3], minval=0, maxval=256, dtype=tf.int32)
    
    # Instantiate generator
    generator = VGG19Generator()
    
    # Load the generator weights from file
    # Base model
    generator.load_weights('models/generator.h5')
    
    # Preprocess the input
    pixels = tf.cast(pixels, tf.float32)
    pixels = tf.expand_dims(pixels, axis=0)
    print(pixels.shape)
    # Generate an image from the input noise vector
    output = generator.predict(pixels)
    output = output * 255
    output_image = output.astype(np.uint8)
    
    # Ensure the generated image has the correct shape
    print(output_image.shape)
    assert output_image.shape == (1, 256, 256, 3)
    
    # Ensure the generated image has no NaN or inf values
    assert not np.isnan(output_image).any()
    assert not np.isinf(output_image).any()
    
    print('Generator test passed.')


if __name__ == '__main__':
    test_generator()