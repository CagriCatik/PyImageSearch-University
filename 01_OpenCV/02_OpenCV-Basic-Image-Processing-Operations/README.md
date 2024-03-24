# Basic Image Processing Operations

## 1. Morphological Operations

```python
# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(image, kernel, iterations=1)
```

## 2. Smoothing and Blurring

```python
# Smoothing
smoothed_image = cv2.blur(image, (5, 5))

# Gaussian Blurring
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
```

## 3. Color Spaces

```python
# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

## 4. Basic Thresholding

```python
# Simple Thresholding
ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
```

## 5. Adaptive Thresholding

```python
# Adaptive Thresholding
adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

## 6. Kernels

```python
# Define a custom kernel
custom_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16

# Apply custom kernel
custom_filtered_image = cv2.filter2D(image, -1, custom_kernel)
```

## 7. Image Gradients

```python
# Sobel Operator
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
```

## 8. Edge Detection

```python
# Canny Edge Detection
edges = cv2.Canny(gray_image, 50, 150)
```

## 9. Automatic Edge Detection

```python
# Laplacian Operator
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
```
