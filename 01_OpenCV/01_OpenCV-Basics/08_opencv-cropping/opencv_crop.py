import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set the default image path
    image_path = "08_opencv-cropping\image\opencv_logo.png"

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
        return

    # Display the original image
    cv2.imshow("Original", image)

    # Crop the face from the image
    face = image[85:250, 85:220]
    cv2.imshow("Face", face)
    cv2.waitKey(0)

    # Crop the body from the image
    body = image[90:450, 0:290]
    cv2.imshow("Body", body)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
