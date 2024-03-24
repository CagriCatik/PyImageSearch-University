
# Basics

## 1. Loading and Displaying Images

```python
import cv2

# Load an image from a file
image = cv2.imread('path/to/image.jpg')

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2. Getting and Setting Pixels

```python
# Accessing pixel values
pixel_value = image[100, 150]

# Modifying pixel values
image[100, 150] = [255, 0, 0]  # Set pixel at (100, 150) to blue
```

## 3. Drawing with OpenCV

```python
# Draw a line
cv2.line(image, (0, 0), (200, 200), (0, 255, 0), 2)

# Draw a rectangle
cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), 3)

# Draw a circle
cv2.circle(image, (100, 100), 20, (255, 255, 0), -1)
```

## 4. Translation

```python
# Define translation matrix
translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])

# Apply translation
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
```

## 5. Rotation

```python
# Define rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 45, 1)

# Apply rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
```

## 6. Resizing

```python
# Resize the image
resized_image = cv2.resize(image, (300, 200))
```

## 7. Flipping

```python
# Flip horizontally
flipped_image = cv2.flip(image, 1)

# Flip vertically
flipped_image = cv2.flip(image, 0)
```

## 8. Cropping

```python
# Crop a region of interest (ROI)
roi = image[50:150, 100:200]
```

## 9. Image Arithmetic

```python
# Add two images
result = cv2.add(image1, image2)

# Subtract two images
result = cv2.subtract(image1, image2)
```

## 10. Bitwise Operations

```python
# Bitwise AND
result = cv2.bitwise_and(image1, image2)

# Bitwise OR
result = cv2.bitwise_or(image1, image2)

# Bitwise XOR
result = cv2.bitwise_xor(image1, image2)

# Bitwise NOT
result = cv2.bitwise_not(image)
```

## 11. Masking

```python
# Create a mask
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)

# Apply the mask
masked_image = cv2.bitwise_and(image, image, mask=mask)
```

## 12. Splitting and Merging Channels

```python
# Split the channels
b, g, r = cv2.split(image)

# Merge the channels
merged_image = cv2.merge([r, g, b])
```
