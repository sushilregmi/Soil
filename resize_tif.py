import os
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import numpy as np


def resize_tif(input_path, output_path, new_size):
    with rasterio.open(input_path) as src:
        img = src.read(1)  # Read the first band
        img_pil = Image.fromarray(img)  # Convert to PIL Image
        img_pil = img_pil.resize(new_size)  # Resize the image
        img_resized = np.array(img_pil)  # Convert back to numpy array

        # Save the resized image as a new TIFF file
        transform = from_origin(src.transform[2], src.transform[5], src.transform[0], src.transform[4])
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=img_resized.shape[0],
            width=img_resized.shape[1],
            count=1,
            dtype=img_resized.dtype,
            crs=src.crs,
            transform=transform,
        ) as dst:
            dst.write(img_resized, 1)

input_directory = 'Dataset/test'
output_directory = 'Dataset/test_new'


os.makedirs(output_directory, exist_ok=True)


new_size = (224, 224)

# Resize and save each TIFF file
for file_name in os.listdir(input_directory):
    if file_name.endswith('.tif'):
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)
        resize_tif(input_path, output_path, new_size)