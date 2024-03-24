import logging
from matplotlib import pyplot as plt
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def grayscale_histogram(image_path):
    # Load the input image and convert it to grayscale
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Unable to load image from path: %s", image_path)
        return

    logger.info("Image loaded successfully.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute a grayscale histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Matplotlib expects RGB images, so convert and then display the image
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])

    # Normalize the histogram
    hist /= hist.sum()

    # Plot the normalized histogram
    plt.figure()
    plt.title("Grayscale Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def main():
    # Path to the image
    image_path = "03_OpenCV-Histograms/01_image-histograms/image/lane.jpg"  # Modify the path accordingly

    # Plot grayscale histogram
    grayscale_histogram(image_path)

if __name__ == "__main__":
    main()
