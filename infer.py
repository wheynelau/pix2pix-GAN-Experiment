from src.models import Generator, VGG19Generator
import argparse
from PIL import Image
import os
import numpy
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser(description='Inference on an image using a saved TensorFlow model.')
parser.add_argument('image_path', type=str, help='path to the input folder')
parser.add_argument('output_path', type=str, help='path to the model folder')

# Testing function
parser.add_argument('-cpu', action='store_true', help='use CPU instead of GPU')
parser.add_argument('-concat', action='store_true', help='concatenate the input and output images')
parser.add_argument('--model_path', type=str, help='path to the model folder, if None use latest time-stamped models',default=None)
args = parser.parse_args()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if not os.path.exists(args.image_path):
    print("Input folder does not exist.")
    exit()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, exist_ok=True)

if args.model_path:
    model_path = os.path.join(args.model_path, 'generator.h5')

else:
    model_folder = os.listdir('models')
    model_path = os.path.join('models', max(model_folder, key=os.path.getctime), 'generator.h5')
try:
    model = VGG19Generator()
    model.load_weights(model_path)  
except:
    model = Generator()
    model.load_weights(model_path)

images = [os.path.join(args.image_path, x) for x in os.listdir(args.image_path)]

for image in tqdm(images):
    input = Image.open(image)
    input = input.convert('RGB')
    input = input.resize((256, 256), Image.BICUBIC)
    input = numpy.array(input)
    input_ = input / 255
    input_ = numpy.expand_dims(input_, axis=0)
    output = model.predict(input_)
    output = output * 255
    output = numpy.squeeze(output, axis=0)
    output_image = Image.fromarray(output.astype(numpy.uint8))
    if args.concat:
        output_image = Image.fromarray(numpy.concatenate((input, output), axis=1).astype(numpy.uint8))
    output_image.save(os.path.join(args.output_path, os.path.basename(image)))

print("Done")