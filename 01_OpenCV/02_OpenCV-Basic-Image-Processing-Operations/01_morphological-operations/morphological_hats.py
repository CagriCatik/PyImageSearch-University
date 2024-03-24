import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def get_concat_h(im1, im2):
    return cv2.hconcat([im1, im2])

# Load the image and display it
image_path = "01_morphological-operations\images\car.png"
logging.info(f"Loading image from: {image_path}")
image = cv2.imread(image_path)
cv2.imshow("Original", image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
logging.info("Converted image to grayscale")

# Construct a rectangular kernel (14x6) and apply blackhat operation
rectangular_kernel = (14, 6)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, rectangular_kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
logging.info("Applied blackhat operation")

# Apply tophat operation
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
logging.info("Applied tophat operation")

# Display the output images side by side
horizontal = get_concat_h(tophat, blackhat)
cv2.imshow("tophat vs blackhat", horizontal)
cv2.imwrite("01_morphological-operations\images\horizontal.jpg", horizontal)
logging.info("Displayed and saved the result image")

cv2.waitKey(0)
cv2.destroyAllWindows()
