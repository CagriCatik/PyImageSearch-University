import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set the default image path
    image_path = "09_opencv-image-arithmetic\image\grand_canyon.png"

    # Show a message about the usage of cv2.add and cv2.subtract
    logger.info("Using cv2.add and cv2.subtract:")

    # Perform cv2.add and cv2.subtract operations
    added = cv2.add(np.uint8([200]), np.uint8([100]))
    subtracted = cv2.subtract(np.uint8([50]), np.uint8([100]))
    logger.info("max of 255: {}".format(added))
    logger.info("min of 0: {}".format(subtracted))

    # Show a message about using NumPy arithmetic operations
    logger.info("\nUsing NumPy arithmetic operations:")

    # Perform NumPy arithmetic operations
    added = np.uint8([200]) + np.uint8([100])
    subtracted = np.uint8([50]) - np.uint8([100])
    logger.info("wrap around: {}".format(added))
    logger.info("wrap around: {}".format(subtracted))

    # Load the original input image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
        return

    # Display the original image
    cv2.imshow("Original", image)

    # Increase pixel intensities by 100
    M = np.ones(image.shape, dtype="uint8") * 100
    added = cv2.add(image, M)
    cv2.imshow("Lighter", added)

    # Decrease pixel intensities by 50
    M = np.ones(image.shape, dtype="uint8") * 50
    subtracted = cv2.subtract(image, M)
    cv2.imshow("Darker", subtracted)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
