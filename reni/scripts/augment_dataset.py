import os
import numpy as np
import imageio

def scale_image(image: np.ndarray, max_brightness: float) -> np.ndarray:
    # Scale pixel values
    max_value = np.nanmax(image)
    scaled_image = (image / max_value) * max_brightness
    return scaled_image

# Define the directory where your dataset is located
dataset_dir = '/workspace/data/RENI_HDR_AUG/train'
output_dir = '/workspace/data/RENI_HDR_SCALE_AUG/train'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all .exr files in the dataset directory
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.exr')]

# Define the range of max brightness values
brightness_values = np.linspace(10, 10000, 10)

for k, image_file in enumerate(image_files):
    # Load the image
    image = imageio.imread(image_file).astype('float32')
    # make any inf values equal to max non inf
    image[image == np.inf] = np.nanmax(image[image != np.inf])
    # make any values less than zero equal to min non negative
    image[image <= 0] = np.nanmin(image[image > 0])

    # Generate augmented images
    for i, brightness in enumerate(brightness_values):
        print(f'Augmenting {k}/{len(image_files)} with brightness {brightness}...')
        scaled_image = scale_image(image, brightness)
        # Save the augmented image
        base = os.path.basename(image_file)
        name, ext = os.path.splitext(base)
        output_file = os.path.join(output_dir, f'{name}_aug_{i}{ext}')
        imageio.imwrite(output_file, scaled_image)
