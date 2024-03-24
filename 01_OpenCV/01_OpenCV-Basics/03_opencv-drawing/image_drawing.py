import cv2
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
    else:
        logger.info("Image loaded successfully.")
    return image

def draw_face_shapes(image):
    cv2.circle(image, (168, 188), 90, (0, 0, 255), 2)
    cv2.circle(image, (150, 164), 10, (0, 0, 255), -1)
    cv2.circle(image, (192, 174), 10, (0, 0, 255), -1)
    cv2.rectangle(image, (134, 200), (186, 218), (0, 0, 255), -1)
    logger.info("Face shapes drawn.")

def show_image(image, window_name="Output"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
        logger.info(f"Directory '{folder_name}' created successfully.")
    except FileExistsError:
        logger.warning(f"Directory '{folder_name}' already exists.")

if __name__ == "__main__":
    logger.info("Processing started.")

    process_folder = "03_opencv-drawing"
    create_folder(process_folder)

    image_path = os.path.join(process_folder, "image", "OpenCV.png")
    image = load_image(image_path)

    if image is not None:
        draw_face_shapes(image)
        show_image(image)

    logger.info("Processing completed.")
