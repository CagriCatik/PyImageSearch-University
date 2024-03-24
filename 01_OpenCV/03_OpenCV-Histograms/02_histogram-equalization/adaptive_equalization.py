import cv2

# Set parameters
image_path = r"03_OpenCV-Histograms\02_histogram-equalization\image\empire_state_cloudy.png"  # Path to the input image
clip_limit = 2.0  # Threshold for contrast limiting
tile_size = 20  # Tile grid size -- divides image into tile x tile cells

# Load the input image from disk and convert it to grayscale
print("[INFO] Loading input image...")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
print("[INFO] Applying CLAHE...")
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
equalized = clahe.apply(gray)

# Show the original grayscale image and CLAHE output image
cv2.imshow("Input", gray)
cv2.imshow("CLAHE", equalized)
cv2.waitKey(0)
