import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_and_display_canvas(canvas, window_name="Canvas"):
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)

def draw_green_line(canvas, start_point, end_point):
    green = (0, 255, 0)
    cv2.line(canvas, start_point, end_point, green)
    logger.info("Green line drawn.")

def draw_red_line(canvas, start_point, end_point, thickness):
    red = (0, 0, 255)
    cv2.line(canvas, start_point, end_point, red, thickness)
    logger.info("Red line drawn.")

def draw_green_square(canvas, top_left, bottom_right):
    green = (0, 255, 0)
    cv2.rectangle(canvas, top_left, bottom_right, green)
    logger.info("Green square drawn.")

def draw_red_rectangle(canvas, top_left, bottom_right, thickness):
    red = (0, 0, 255)
    cv2.rectangle(canvas, top_left, bottom_right, red, thickness)
    logger.info("Red rectangle drawn.")

def draw_blue_filled_rectangle(canvas, top_left, bottom_right):
    blue = (255, 0, 0)
    cv2.rectangle(canvas, top_left, bottom_right, blue, -1)
    logger.info("Blue filled rectangle drawn.")

def draw_white_circles(canvas, center, radii_increment):
    white = (255, 255, 255)
    for r in range(0, 175, radii_increment):
        cv2.circle(canvas, center, r, white)
    logger.info("White circles drawn.")

def draw_random_circles(canvas, num_circles):
    for i in range(num_circles):
        radius = np.random.randint(5, high=200)
        color = np.random.randint(0, high=256, size=(3,)).tolist()
        pt = np.random.randint(0, high=300, size=(2,))
        cv2.circle(canvas, tuple(pt), radius, color, -1)
    logger.info(f"{num_circles} random circles drawn.")

if __name__ == "__main__":
    logger.info("Drawing started.")

    canvas = np.zeros((300, 300, 3), dtype="uint8")

    draw_green_line(canvas, (0, 0), (300, 300))
    draw_and_display_canvas(canvas)

    draw_red_line(canvas, (300, 0), (0, 300), 3)
    draw_and_display_canvas(canvas)

    draw_green_square(canvas, (10, 10), (60, 60))
    draw_and_display_canvas(canvas)

    draw_red_rectangle(canvas, (50, 200), (200, 225), 5)
    draw_and_display_canvas(canvas)

    draw_blue_filled_rectangle(canvas, (200, 50), (225, 125))
    draw_and_display_canvas(canvas)

    canvas = np.zeros((300, 300, 3), dtype="uint8")
    draw_white_circles(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2), 25)
    draw_and_display_canvas(canvas)

    canvas = np.zeros((300, 300, 3), dtype="uint8")
    draw_random_circles(canvas, 25)
    draw_and_display_canvas(canvas)

    logger.info("Drawing completed.")
