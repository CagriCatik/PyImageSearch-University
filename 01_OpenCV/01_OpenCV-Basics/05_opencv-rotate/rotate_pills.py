import numpy as np
import cv2
import logging

# Configure logging
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

def convert_to_grayscale(image):
    gray_converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./image/grayscale.jpg", gray_converted)
    logger.info("Image converted to grayscale.")
    cv2.imshow("Grayscale", gray_converted)
    cv2.waitKey(0)
    return gray_converted

def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imwrite("./image/blurred.jpg", blurred_image)
    logger.info("Gaussian blur applied.")
    cv2.imshow("Gaussian blur", blurred_image)
    cv2.waitKey(0)
    return blurred_image

def perform_canny_edge_detection(image):
    edged = cv2.Canny(image, 20, 100)
    cv2.imwrite("./image/canny.jpg", edged)
    logger.info("Canny edge detection applied.")
    cv2.imshow("Canny edge", edged)
    cv2.waitKey(0)
    return edged

def find_and_draw_contours(edged_image):
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(edged_image.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(c)
        image_roi = image[y:y + h, x:x + w]
        mask_roi = mask[y:y + h, x:x + w]
        image_roi = cv2.bitwise_and(image_roi, image_roi, mask=mask_roi)
        logger.info("Contours found and mask applied.")
        return image_roi, (w, h)

def rotate_and_display_images(image_roi, dimensions):
    for angle in range(0, 360, 15):
        rotated = cv2.warpAffine(image_roi, cv2.getRotationMatrix2D((dimensions[0] // 2, dimensions[1] // 2), angle, 1.0), (dimensions[0], dimensions[1]))
        cv2.imshow("Rotated (Problematic)", rotated)
        logger.info(f"Image rotated (Problematic) at angle {angle} degrees.")
        cv2.waitKey(0)

    for angle in range(0, 360, 15):
        rotated = cv2.warpAffine(image_roi, cv2.getRotationMatrix2D((dimensions[0] // 2, dimensions[1] // 2), angle, 1.0), (dimensions[0], dimensions[1]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        cv2.imshow("Rotated (Correct)", rotated)
        logger.info(f"Image rotated (Correct) at angle {angle} degrees.")
        cv2.waitKey(0)

if __name__ == "__main__":
    image_path = "05_opencv-rotate\image\pill_01.png"
    logger.info("Processing started.")

    image = load_and_display_image(image_path)
    if image is not None:
        gray_converted = convert_to_grayscale(image)
        blurred_image = apply_gaussian_blur(gray_converted)
        edged_image = perform_canny_edge_detection(blurred_image)
        image_roi, dimensions = find_and_draw_contours(edged_image)
        if image_roi is not None:
            rotate_and_display_images(image_roi, dimensions)

    logger.info("Processing completed.")
