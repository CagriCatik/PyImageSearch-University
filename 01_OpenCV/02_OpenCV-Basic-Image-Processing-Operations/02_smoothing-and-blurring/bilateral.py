import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)

def apply_bilateral_filter(image, diameter, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def display_all_images(images, window_title):
    concatenated_image = cv2.hconcat(images)
    cv2.imshow(window_title, concatenated_image)
    cv2.waitKey(0)

def main():
    # Load the image and display it
    image_path = "02_smoothing-and-blurring/image/car.png"
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    display_image("Original", image)

    # Construct a list of bilateral filtering parameters
    params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

    processed_images = []
    # Loop over the diameter, sigma color, and sigma space
    for (diameter, sigmaColor, sigmaSpace) in params:
        # Apply bilateral filtering to the image using the current set of parameters
        blurred = apply_bilateral_filter(image, diameter, sigmaColor, sigmaSpace)

        # Show the output image and associated parameters
        title = f"Blurred d={diameter}, sc={sigmaColor}, ss={sigmaSpace}"
        display_image(title, blurred)
        logging.info(f"Applied bilateral filter with parameters: {title}")

        processed_images.append(blurred)

    # Display all processed images in the same window
    display_all_images(processed_images, "Processed Images")

if __name__ == "__main__":
    main()
