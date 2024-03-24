# OpenCV Histograms

**Image Histograms**

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('image.jpg', 0)

# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0,256])

# Plot histogram
plt.plot(hist, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Image Histogram')
plt.show()
```

**Histogram and Adaptive Histogram Equalization**

```python
# Applying histogram equalization
equ = cv2.equalizeHist(img)

# Applying adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)
```

**Histogram Matching**

```python
# Load reference and source images
ref_img = cv2.imread('ref_image.jpg', 0)
src_img = cv2.imread('src_image.jpg', 0)

# Calculate histograms
ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0,256])
src_hist = cv2.calcHist([src_img], [0], None, [256], [0,256])

# Apply histogram matching
matched_img = cv2.calcHist([src_img], [0], None, [256], [0,256])
```

**Gamma Correction**

```python
# Define gamma correction function
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Apply gamma correction
gamma_corrected_img = gamma_correction(img, gamma=1.5)
```

**Automatic Color Correction**

```python
# Load color image
color_img = cv2.imread('color_image.jpg')

# Convert BGR to LAB color space
lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)

# Perform automatic color correction
lab_planes = cv2.split(lab_img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab_img = cv2.merge(lab_planes)
auto_color_corrected_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
```

**Detecting Low Contrast Images**

```python
# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0,256])

# Check for low contrast
total_pixels = img.shape[0] * img.shape[1]
low_contrast_pixels = np.sum(hist < 0.05 * total_pixels)
if low_contrast_pixels > 0.2 * total_pixels:
    print("Image has low contrast.")
else:
    print("Image has sufficient contrast.")
```
