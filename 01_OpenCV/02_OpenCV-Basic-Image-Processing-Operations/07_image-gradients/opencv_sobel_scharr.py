import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def calculate_sobel_scharr(image_path, use_scharr=False):
    # Load the image, convert it to grayscale, and display the original grayscale image
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Gray", gray)

    # Set the kernel size, depending on whether we are using the Sobel or the Scharr operator
    ksize = -1 if use_scharr else 3

    # Compute the gradients along the x and y axis, respectively
    gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

    # The gradient magnitude images are now of the floating-point data type,
    # so we need to convert them back to unsigned 8-bit integer representation
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    # Combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    # Show the output images
    display_image("Sobel/Scharr X", gX)
    display_image("Sobel/Scharr Y", gY)
    display_image("Sobel/Scharr Combined", combined)

def main():
    # Specify the path to the input image
    image_path = "07_image-gradients\image\_bricks.png"
    calculate_sobel_scharr(image_path)

if __name__ == "__main__":
    main()
