import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)

def otsu_thresholding(image_path):
    # Load the image and display it
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    display_image("Image", image)

    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Otsu's automatic thresholding
    (T, threshInv) = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    logging.info(f"Otsu's thresholding value: {T}")

    # Visualize only the masked regions in the image
    masked = cv2.bitwise_and(image, image, mask=threshInv)

    # Display the thresholded image
    display_image("Threshold", threshInv)

    # Display the output with masked regions
    display_image("Output", masked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Specify the image path
    image_path = "04_basic-thresholding\images\coins01.png"
    otsu_thresholding(image_path)

if __name__ == "__main__":
    main()
