import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Unable to load the image from {image_path}")
    return image


def resize_and_show(image, target_size, method_name):
    r = target_size / image.shape[1] if method_name == "Width" else target_size / image.shape[0]
    dim = (target_size, int(image.shape[0] * r)) if method_name == "Width" else (int(image.shape[1] * r), target_size)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(f"Resized ({method_name})", resized)
    cv2.waitKey(0)


def main():
    image_path = "06_opencv-resizing/image/pill_01.png"
    image = load_image(image_path)
    if image is None:
        return

    cv2.imshow("Original", image)

    resize_and_show(image, 150, "Width")
    resize_and_show(image, 50, "Height")
    resize_and_show(image, 100, "OpenCV")

    methods = [
        ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
        ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
        ("cv2.INTER_AREA", cv2.INTER_AREA),
        ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
        ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)
    ]

    for (name, method) in methods:
        logger.info(f"[INFO] {name}")
        resized = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3), interpolation=method)
        cv2.imshow(f"Method: {name}", resized)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
