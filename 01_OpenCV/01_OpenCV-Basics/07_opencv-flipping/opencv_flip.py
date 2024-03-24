import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)


def main():
 
    # Load the image and display it to our screen
    image_path = "07_opencv-flipping\image\opencv_logo.png"
    image = cv2.imread(image_path)

    if image is None:
        logging.error(f"Unable to load image from path: {image_path}")
        return

    show_and_wait("Original", image)

    # Flip the image horizontally
    flipped_horizontally = cv2.flip(image, 1)
    process_and_show(flipped_horizontally, "horizontally-flipped.jpg", "horizontally")

    # Flip the image vertically
    flipped_vertically = cv2.flip(image, 0)
    process_and_show(flipped_vertically, "vertically-flipped.jpg", "vertically")

    # Flip the image along both axes
    flipped_both = cv2.flip(image, -1)
    process_and_show(flipped_both, "horizontally-vertically-flipped.jpg", "horizontally and vertically")

def show_and_wait(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

def process_and_show(img, output_filename, flip_type):
    logging.info(f"Flipping image {flip_type}...")
    show_and_wait(f"Flipped {flip_type.capitalize()}", img)
    output_path = f"./images_solution/{output_filename}"
    cv2.imwrite(output_path, img)
    logging.info(f"Result saved as {output_path}")

if __name__ == "__main__":
    main()
