from src.models import Generator, VGG19Generator
from PIL import Image
import os
import numpy
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def infer(args: DictConfig):
    if args.infer.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(args.infer.image_folder):
        print("Input folder does not exist.")
        exit()

    if not os.path.exists(args.infer.output_folder):
        os.makedirs(args.infer.output_folder, exist_ok=True)

    try:
        model_path = os.path.join(args.infer.model_path, 'generator.h5')
    except Exception as e:
        print("Model path does not exist.")
        exit()
    try:
        model = VGG19Generator()
        model.load_weights(model_path)  
    except:
        model = Generator()
        model.load_weights(model_path)

    images = [os.path.join(args.infer.image_folder, x) for x in os.listdir(args.infer.image_folder)]

    for image in tqdm(images):
        input = Image.open(image)
        input = input.convert('RGB')
        input = input.resize((args.infer.size, args.infer.size), Image.BICUBIC)
        input = numpy.array(input)
        input_ = input / 255
        input_ = numpy.expand_dims(input_, axis=0)
        output = model.predict(input_)
        output = output * 255
        output = numpy.squeeze(output, axis=0)
        output_image = Image.fromarray(output.astype(numpy.uint8))
        if args.infer.concat:
            output_image = Image.fromarray(numpy.concatenate((input, output), axis=1).astype(numpy.uint8))
        output_image.save(os.path.join(args.infer.output_folder, os.path.basename(image)))

    print("Done")

if __name__ == "__main__":
    infer()