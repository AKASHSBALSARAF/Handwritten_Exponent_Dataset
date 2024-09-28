# main_combiner.py

import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image

# Directories
exponent_dir = r"C:\Users\akash\Documents\GitHub\Handwritten_Exponent_Dataset\data\EXPONENT"
output_dir = r"C:\Users\akash\Documents\GitHub\Handwritten_Exponent_Dataset\data\OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (_, _) = mnist

# Parameters
num_images_to_generate = 10
base_size = (28, 28)
exponent_size = (12, 12)
base_position = (15, 22)  # Center position for the base image
exponent_position_offset = (25, 0)  # Offset for the exponent position relative to the base

# Function to resize images
def resize_image(image, target_size):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(target_size, Image.LANCZOS)
    return np.array(resized_image)

# Function to combine base and exponent images
def combine_images(base_image, exponent_image, base_position, exponent_position_offset):
    # Create a blank canvas
    canvas_size = (64, 64)
    canvas = Image.new('L', canvas_size, color=0)  # Black background
    
    # Convert to PIL images
    base_image_pil = Image.fromarray(base_image)
    exponent_image_pil = Image.fromarray(exponent_image)
    
    # Calculate exponent position
    exponent_position = (base_position[0] + exponent_position_offset[0], 
                         base_position[1] + exponent_position_offset[1])
    
    # Paste images on the canvas
    canvas.paste(base_image_pil, base_position)
    canvas.paste(exponent_image_pil, exponent_position)
    
    return np.array(canvas)

# Prepare a list to collect image data for CSV
image_data = []

# Generate the dataset
def generate_combined_images(mnist_images, mnist_labels, exponent_dir, output_dir, num_images_to_generate, base_size, exponent_size, base_position, exponent_position_offset):
    # Get the list of exponent image files
    exponent_files = [f for f in os.listdir(exponent_dir) if f.endswith('.png')]
    
    for i in range(num_images_to_generate):
        base_idx = np.random.randint(0, len(mnist_images))
        base_image = mnist_images[base_idx]
        base_label = mnist_labels[base_idx]

        exponent_file = np.random.choice(exponent_files)
        exponent_image_path = os.path.join(exponent_dir, exponent_file)
        exponent_image = cv2.imread(exponent_image_path, cv2.IMREAD_GRAYSCALE)

        # Resize images
        base_image = resize_image(base_image, base_size)
        exponent_image = resize_image(exponent_image, exponent_size)

        # Combine images
        combined_image = combine_images(base_image, exponent_image, base_position, exponent_position_offset)
        
        # Save the combined image
        combined_image_name = f'img_{i+1}.png'
        combined_image_path = os.path.join(output_dir, combined_image_name)
        cv2.imwrite(combined_image_path, combined_image)
        
        # Record image data
        image_data.append({
            'base': base_label,
            'exponent': exponent_file,
            'position': combined_image_path
        })

# Generate the combined images
generate_combined_images(train_images, train_labels, exponent_dir, output_dir, num_images_to_generate, base_size, exponent_size, base_position, exponent_position_offset)

# Create a DataFrame and save to CSV
image_df = pd.DataFrame(image_data)
csv_path = os.path.join(output_dir, 'combined_image_data.csv')
image_df.to_csv(csv_path, index=False)

print(f'Images saved to {output_dir}')
print(f'CSV file saved to {csv_path}')
