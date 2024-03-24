import cv2
import logging

logging.basicConfig(level=logging.DEBUG)  # Setze das Logging-Level auf DEBUG

def main(image_path="02_opencv-getting-setting\image\OpenCV.png"):
    # load the image, grab its spatial dimensions (width and height),
    # and then display the original image to our screen
    image = cv2.imread(image_path)
    
    if image is None:
        logging.error(f"Unable to load image from path: {image_path}")
        return

    h, w, _ = image.shape
    cv2.imshow("Original", image)

    # access the pixel located at x=50, y=20
    print_pixel_info(image, 0, 0, "Top-Left Corner")

    # update the pixel at (50, 20) and set it to red
    update_pixel(image, 20, 50, (0, 0, 255))
    print_pixel_info(image, 20, 50, "Updated Pixel")

    # compute the center of the image, which is simply the width and height
    # divided by two
    cX, cY = w // 2, h // 2

    # crop and display image corners
    crop_and_display(image, 0, cX, 0, cY, "Top-Left Corner")
    crop_and_display(image, cX, w, 0, cY, "Top-Right Corner")
    crop_and_display(image, cX, w, cY, h, "Bottom-Right Corner")
    crop_and_display(image, 0, cX, cY, h, "Bottom-Left Corner")

    # set the top-left corner of the original image to be green
    update_pixel(image, 0, 0, (0, 255, 0))

    # Show our updated image
    cv2.imshow("Updated", image)
    cv2.waitKey(0)

def print_pixel_info(image, x, y, title):
    b, g, r = image[y, x]
    logging.info(f"{title} - Red: {r}, Green: {g}, Blue: {b}")

def update_pixel(image, x, y, color):
    image[y, x] = color

def crop_and_display(image, x1, x2, y1, y2, title):
    cropped = image[y1:y2, x1:x2]
    cv2.imshow(title, cropped)

if __name__ == "__main__":
    main()