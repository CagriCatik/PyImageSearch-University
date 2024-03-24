import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def canny_edge_detection(image_path):
    # Load the image, convert it to grayscale, and blur it slightly
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)

    # Display the original image
    display_image("Original", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection using a wide threshold, tight threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)

    # Concatenate the edge images horizontally
    edges_combined = np.hstack([wide, tight, auto])

    # Resize the concatenated image to make it smaller
    scale_percent = 50  # percent of original size
    width = int(edges_combined.shape[1] * scale_percent / 100)
    height = int(edges_combined.shape[0] * scale_percent / 100)
    edges_combined_resized = cv2.resize(edges_combined, (width, height))

    # Show the resized edges figure
    display_image("Edges", edges_combined_resized)

def main():
    # Specify the path to the input image
    image_path = "image/lane.jpg"
    canny_edge_detection(image_path)

if __name__ == "__main__":
    main()
