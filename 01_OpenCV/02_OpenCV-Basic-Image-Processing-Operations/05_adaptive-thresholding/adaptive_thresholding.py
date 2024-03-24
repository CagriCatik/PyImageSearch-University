import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def adaptive_thresholding(image_path):
    # Load the image and display it
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    display_image("Image", image)

    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply simple thresholding with a hardcoded threshold value
    (T, threshInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    display_image("Simple Thresholding", threshInv)

    # Apply Otsu's automatic thresholding
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    display_image("Otsu Thresholding", threshInv)

    # Adaptive thresholding using mean
    thresh_mean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    display_image("Mean Adaptive Thresholding", thresh_mean)

    # Adaptive thresholding using Gaussian weighting
    thresh_gaussian = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    display_image("Gaussian Adaptive Thresholding", thresh_gaussian)

def main():
    # Specify the image path
    image_path = "05_adaptive-thresholding\image\steve_jobs.png"
    adaptive_thresholding(image_path)

if __name__ == "__main__":
    main()
