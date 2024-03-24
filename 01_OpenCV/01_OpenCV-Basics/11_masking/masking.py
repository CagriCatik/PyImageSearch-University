import numpy as np
import cv2

def apply_mask(image, mask):
    """
    Apply the provided mask to the image using bitwise_and operation.
    """
    return cv2.bitwise_and(image, image, mask=mask)

# Load the original input image and display it
image_path = "11_masking/image/opencv_logo.png"
image = cv2.imread(image_path)
cv2.imshow("Original", image)

# Create a rectangular mask
rectangular_mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.rectangle(rectangular_mask, (0, 90), (290, 450), 255, -1)
cv2.imshow("Rectangular Mask", rectangular_mask)

# Apply the rectangular mask
masked_rectangular = apply_mask(image, rectangular_mask)
cv2.imshow("Rectangular Mask Applied", masked_rectangular)

# Create a circular mask
circular_mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(circular_mask, (145, 200), 100, 255, -1)
cv2.imshow("Circular Mask", circular_mask)

# Apply the circular mask
masked_circular = apply_mask(image, circular_mask)
cv2.imshow("Circular Mask Applied", masked_circular)

cv2.waitKey(0)
