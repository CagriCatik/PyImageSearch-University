import cv2

# Set input image path
image_path = r"03_OpenCV-Histograms\02_histogram-equalization\image\empire_state_cloudy.png"

# Load the input image from disk and convert it to grayscale
print("[INFO] Loading input image...")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
print("[INFO] Performing histogram equalization...")
equalized = cv2.equalizeHist(gray)

# Show the original grayscale image and equalized image
cv2.imshow("Input", gray)
cv2.imshow("Histogram Equalization", equalized)
cv2.waitKey(0)
