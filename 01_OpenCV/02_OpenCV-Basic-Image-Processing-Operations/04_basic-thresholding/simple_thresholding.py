import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)

def simple_thresholding(image_path):
    # Load the image and display it
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    display_image("Image", image)

    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply basic thresholding (inverse)
    (T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    display_image("Threshold Binary Inverse", threshInv)

    # Apply basic thresholding (normal)
    (T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    display_image("Threshold Binary", thresh)

    # Visualize only the masked regions in the image
    masked = cv2.bitwise_and(image, image, mask=threshInv)
    display_image("Output", masked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Specify the image path
    image_path = "04_basic-thresholding\images\coins01.png"
    simple_thresholding(image_path)

if __name__ == "__main__":
    main()
