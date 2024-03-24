import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()

def calculate_magnitude_and_orientation(image_path):
    # Load the input image and convert it to grayscale
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients along the x and y axis, respectively
    gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

    # Display the images
    display_image("Grayscale", gray, cmap="gray")
    display_image("Gradient Magnitude", magnitude, cmap="jet")
    display_image("Gradient Orientation [0, 180]", orientation, cmap="jet")

def main():
    # Specify the path to the input image
    image_path = "07_image-gradients\image\_bricks.png"
    calculate_magnitude_and_orientation(image_path)

if __name__ == "__main__":
    main()
