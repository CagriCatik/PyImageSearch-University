import cv2
import logging

logging.basicConfig(level=logging.INFO)

def apply_blur(image, kernel_size, blur_type):
    if blur_type == "average":
        return cv2.blur(image, kernel_size)
    elif blur_type == "gaussian":
        return cv2.GaussianBlur(image, kernel_size, 0)
    elif blur_type == "median":
        return cv2.medianBlur(image, kernel_size[0])
    else:
        raise ValueError("Invalid blur type")

def display_all_images(images, window_title):
    concatenated_image = cv2.hconcat(images)
    cv2.imshow(window_title, concatenated_image)
    cv2.waitKey(0)

def main():
    # construct the argument parser and parse the arguments
    image_path = "02_smoothing-and-blurring\image\car.png"
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    cv2.imshow("Original", image)

    kernelSizes = [(3, 3), (9, 9), (15, 15)]
    blur_types = ["average", "gaussian", "median"]

    processed_images = []

    # loop over the blur types and kernel sizes
    for blur_type in blur_types:
        for kernel_size in kernelSizes:
            # apply the specified blur to the image
            blurred = apply_blur(image, kernel_size, blur_type)

            # show the output image and associated parameters
            title = f"{blur_type.capitalize()} ({kernel_size[0]}, {kernel_size[1]})"
            display_all_images([blurred], title)
            logging.info(f"Applied {blur_type} blur with kernel size {kernel_size}")

            processed_images.append(blurred)

    # Display all processed images in the same window
    display_all_images(processed_images, "Processed Images")

if __name__ == "__main__":
    main()
