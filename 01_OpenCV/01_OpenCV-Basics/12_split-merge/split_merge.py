import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set the default image path
    image_path = "12_split-merge\image\opencv_logo.png"

    # Load the input image and grab each channel
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
        return

    (B, G, R) = cv2.split(image)

    # Show each channel individually
    cv2.imshow("Red", R)
    cv2.imshow("Green", G)
    cv2.imshow("Blue", B)
    cv2.waitKey(0)

    # Merge the image back together again
    merged = cv2.merge([B, G, R])
    cv2.imshow("Merged", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize each channel in color
    zeros = np.zeros(image.shape[:2], dtype="uint8")
    cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
    cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
    cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
