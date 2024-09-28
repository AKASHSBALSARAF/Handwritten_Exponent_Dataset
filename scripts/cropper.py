# main_cropper.py

import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd

# Directory to save the cropped images
save_dir = r"C:\Users\akash\Documents\GitHub\Handwritten_Exponent_Dataset\data\EXPONENT"
os.makedirs(save_dir, exist_ok=True)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (_, _) = mnist

# Function to remove the background and create an alpha channel for transparency
def remove_background_and_create_alpha(image):
    # Apply adaptive thresholding to create a binary mask
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphology to refine the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours to crop the image tightly around the digit
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # If no contours found, return a fully transparent image
        h, w = image.shape
        return np.zeros((h, w, 4), dtype=np.uint8)
    
    # Combine all contours to ensure the entire digit is captured
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Resize cropped image and mask to a standard size
    standard_size = (28, 28)  # Resize to the original MNIST size
    resized_image = cv2.resize(cropped_image, standard_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, standard_size, interpolation=cv2.INTER_LINEAR)

    # Create alpha channel from the resized mask
    alpha_channel = resized_mask
    alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), sigmaX=2, sigmaY=2)
    alpha_channel = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)  # Ensure binary alpha channel
    
    # Create BGRA image
    result = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGRA)
    result[:, :, 3] = alpha_channel

    # Ensure no black areas remain by replacing any black (0,0,0) with transparent
    result[np.all(result[:, :, :3] == (0, 0, 0), axis=-1)] = (0, 0, 0, 0)

    return result

# Prepare a list to collect image data for CSV
image_data = []

# Define the number of images to generate per digit
num_images_per_digit = 20  # Adjust this number as needed

# Save images for each digit (0-9)
def save_digit_images(images, labels, save_dir, num_images_per_digit):
    digit_count = {i: 0 for i in range(10)}
    
    for img, label in zip(images, labels):
        if digit_count[label] < num_images_per_digit:
            alpha_image = remove_background_and_create_alpha(img)
            label_name = chr(97 + label)  # Convert digit to corresponding label (e.g., 0 -> 'a')
            filename = f'{label_name}{digit_count[label] + 1}.png'
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, alpha_image)
            image_data.append({'position': filepath, 'number': label})
            digit_count[label] += 1

        if all(count == num_images_per_digit for count in digit_count.values()):  # All digits have the required number of images
            break

# Preprocess and save the images
save_digit_images(train_images, train_labels, save_dir, num_images_per_digit)

# Create a DataFrame and save to CSV
image_df = pd.DataFrame(image_data)
csv_path = os.path.join(save_dir, 'image_data.csv')
image_df.to_csv(csv_path, index=False)

print(f'Images saved to {save_dir}')
print(f'CSV file saved to {csv_path}')
