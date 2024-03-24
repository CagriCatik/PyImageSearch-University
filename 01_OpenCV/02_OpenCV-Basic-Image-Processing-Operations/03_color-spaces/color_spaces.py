import cv2
import logging

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)

def display_all_channels(image, space_name, channel_names, space_conversion):
    converted_image = cv2.cvtColor(image, space_conversion)
    display_image(space_name, converted_image)

    for (name, chan) in zip(channel_names, cv2.split(converted_image)):
        display_image(name, chan)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # construct the argument parser and parse the arguments
    image_path = "03_color-spaces\image\car.png"
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)

    # Display RGB channels
    display_all_channels(image, "RGB", ("B", "G", "R"), cv2.COLOR_BGR2RGB)

    # Display HSV channels
    display_all_channels(image, "HSV", ("H", "S", "V"), cv2.COLOR_BGR2HSV)

    # Display L*a*b* channels
    display_all_channels(image, "L*a*b*", ("L*", "a*", "b*"), cv2.COLOR_BGR2Lab)

    # Display Grayscale version
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Original", image)
    display_image("Grayscale", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
