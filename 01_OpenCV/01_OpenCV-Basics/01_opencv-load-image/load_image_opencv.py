import logging
from matplotlib import pyplot as plt
import cv2
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plt_imshow(title, image):
    # convert the image from BGR to RGB color space and display it
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')  # Turn off axis
    plt.show()

def main():
    # Hard-coded image path
    image_path = "01_opencv-load-image/image/OpenCV.png"

    # Check if image path exists
    if not os.path.exists(image_path):
        logger.error(f"Image file '{image_path}' not found.")
        return

    # Load the image from disk via "cv2.imread"
    image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
        return

    # Get the spatial dimensions of the image
    (h, w, c) = image.shape

    # Display image information
    logger.info("Width: %d pixels", w)
    logger.info("Height: %d pixels", h)
    logger.info("Channels: %d", c)

    # Show the image
    plt_imshow("Image", image)

if __name__ == "__main__":
    main()
