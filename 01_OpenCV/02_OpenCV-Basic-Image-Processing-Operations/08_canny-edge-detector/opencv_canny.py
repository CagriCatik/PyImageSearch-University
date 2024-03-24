import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def find_optimal_canny_threshold(image, low_start=10, high_start=200, step=10):
    optimal_low = low_start
    optimal_high = high_start
    optimal_edges = cv2.Canny(image, optimal_low, optimal_high)

    for low in range(low_start, 256, step):
        for high in range(high_start, 256, step):
            edges = cv2.Canny(image, low, high)
            if cv2.countNonZero(edges) < cv2.countNonZero(optimal_edges):
                optimal_low = low
                optimal_high = high
                optimal_edges = edges

    return optimal_low, optimal_high, optimal_edges

def canny_edge_detection(image_path):
    # Load the image, convert it to grayscale, and blur it slightly
    logging.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    display_image("Original", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Show the original and blurred images
    display_image("Blurred", blurred)

    # Find optimal Canny thresholds
    low, high, wide_edges = find_optimal_canny_threshold(blurred)
    logging.info(f"Optimal thresholds for wide edges: Low={low}, High={high}")
    display_image("Wide Edge Map", wide_edges)

    low, high, mid_edges = find_optimal_canny_threshold(blurred, low_start=30, high_start=150)
    logging.info(f"Optimal thresholds for mid-range edges: Low={low}, High={high}")
    display_image("Mid Edge Map", mid_edges)

    low, high, tight_edges = find_optimal_canny_threshold(blurred, low_start=240, high_start=250)
    logging.info(f"Optimal thresholds for tight edges: Low={low}, High={high}")
    display_image("Tight Edge Map", tight_edges)

def main():
    # Get the absolute path to the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "image", "lane.jpg")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        logging.error("Image file not found.")
        return

    print("Image path:", image_path)
    canny_edge_detection(image_path)




if __name__ == "__main__":
    main()
