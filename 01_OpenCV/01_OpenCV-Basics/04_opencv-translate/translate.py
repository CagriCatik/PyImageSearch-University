import numpy as np
import cv2
import os
import imutils

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_and_display_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
    else:
        logger.info("Image loaded successfully.")
        cv2.imshow("Original", image)
        cv2.waitKey(0)
    return image

def shift_image(image, shift_x, shift_y, output_path):
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv2.imshow("Shifted Image", shifted)
    cv2.imwrite(output_path, shifted)
    cv2.waitKey(0)
    return shifted

def translate_image_imutils(image, shift_x, shift_y, output_path):
    shifted = imutils.translate(image, shift_x, shift_y)
    cv2.imshow("Translated Image (imutils)", shifted)
    cv2.imwrite(output_path, shifted)
    cv2.waitKey(0)
    return shifted

if __name__ == "__main__":
    logger.info("Processing started.")

    image_path = "04_opencv-translate\image\opencv_logo.png"
    image = load_and_display_image(image_path)

    if image is not None:
        # Shift the image 25 pixels to the right and 50 pixels down
        shift_image(image, 25, 50, "04_opencv-translate\image\shifted-down-right.jpg")

        # Shift the image 50 pixels to the left and 90 pixels up
        shift_image(image, -50, -90, "04_opencv-translate\image\shifted-up-left.jpg")

        # Use imutils to translate the image 100 pixels down
        translate_image_imutils(image, 0, 100, "04_opencv-translate\image\shifted-down.jpg")

    logger.info("Processing completed.")
