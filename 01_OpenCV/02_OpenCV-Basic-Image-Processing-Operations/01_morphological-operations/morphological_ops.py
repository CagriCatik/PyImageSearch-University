import cv2
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

def get_concat_h(*images):
    return cv2.hconcat(images)

def get_concat_v(*images):
    return cv2.vconcat(images)

def load_image(image_path):
    logging.info(f"Loading image from: {image_path}")
    return cv2.imread(image_path)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def perform_morphological_operation(image, operation, kernel_size, iterations, save_path=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    result = operation(image, None, iterations=iterations)
    if save_path:
        save_image(result, save_path)
    return result

# Load the image and display it
image_path = "01_morphological-operations/images/car.png"
image = load_image(image_path)
display_image("Original", image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
logging.info("Converted image to grayscale")

# Erosion operation
erosion_iterations = [1, 2, 3, 4]
eroded_images = [perform_morphological_operation(gray, cv2.erode, (3, 3), i) for i in erosion_iterations]
horizontal_eroded = get_concat_h(*eroded_images)
display_image("Eroded", horizontal_eroded)
save_image(horizontal_eroded, "01_morphological-operations/images/eroded_images.jpg")
logging.info("Performed erosion operation")

# Dilation operation
dilation_iterations = [1, 2, 3, 4]
dilated_images = [perform_morphological_operation(gray, cv2.dilate, (3, 3), i) for i in dilation_iterations]
horizontal_dilated = get_concat_h(*dilated_images)
display_image("Dilated", horizontal_dilated)
save_image(horizontal_dilated, "01_morphological-operations/images/dilated_images.jpg")
logging.info("Performed dilation operation")

# Compare eroded and dilated images vertically
vertical_compare = get_concat_v(horizontal_eroded, horizontal_dilated)
display_image("4 times eroded and dilated", vertical_compare)
cv2.waitKey(0)

# Opening operation
opening_kernel_size = (5, 5)
opening = perform_morphological_operation(gray, cv2.morphologyEx, opening_kernel_size, 1, "01_morphological-operations/images/opening.jpg")
display_image("Opening", opening)
logging.info("Performed opening operation")

# Closing operation
closing_kernel_size = (5, 5)
closing = perform_morphological_operation(gray, cv2.morphologyEx, closing_kernel_size, 1, "01_morphological-operations/images/closing.jpg")
display_image("Closing", closing)
logging.info("Performed closing operation")

# Gradient operation
gradient_kernel_size = (5, 5)
gradient = perform_morphological_operation(gray, cv2.morphologyEx, gradient_kernel_size, 1, "01_morphological-operations/images/gradient.jpg")
display_image("Gradient", gradient)
logging.info("Performed gradient operation")

cv2.destroyAllWindows()
