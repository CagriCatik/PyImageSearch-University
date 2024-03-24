from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)

def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

def apply_kernels(image, kernelBank):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for (kernelName, kernel) in kernelBank:
        logging.info(f"Applying {kernelName} kernel")
        convoleOutput = convolve(gray, kernel)
        opencvOutput = cv2.filter2D(gray, -1, kernel)

        display_image("original", gray)
        display_image(f"{kernelName} - convole", convoleOutput)
        display_image(f"{kernelName} - opencv", opencvOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Specify the image path
    image_path = "06_convolutions-opencv\image\pokemon.png"
    
    # Construct average blurring kernels used to smooth an image
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

    # Construct a sharpening filter
    sharpen = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")

    # Construct the Laplacian kernel used to detect edge-like regions of an image
    laplacian = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="int")

    # Construct the Sobel x-axis kernel
    sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="int")

    # Construct the Sobel y-axis kernel
    sobelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="int")

    # Construct the kernel bank
    kernelBank = [
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY)
    ]

    # Load the input image
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)

    # Apply kernels
    apply_kernels(image, kernelBank)

if __name__ == "__main__":
    main()
