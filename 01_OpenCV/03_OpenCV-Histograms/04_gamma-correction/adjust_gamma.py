import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Set input image path
image_path = r"03_OpenCV-Histograms\04_gamma-correction\example.png"

# Load the original image
original = cv2.imread(image_path)

# Loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
    # Ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        continue

    # Apply gamma correction and show the images
    gamma = max(0.1, gamma)  # Ensuring gamma is not less than 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey(0)
