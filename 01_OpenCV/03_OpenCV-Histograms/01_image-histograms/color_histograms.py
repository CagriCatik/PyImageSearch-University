import logging
from matplotlib import pyplot as plt
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_histograms(image):
    # split the image into its respective channels
    blue_channel, green_channel, red_channel = cv2.split(image)
    channels = (blue_channel, green_channel, red_channel)
    colors = ("b", "g", "r")

    # initialize the figure for plotting
    plt.figure()

    # loop over the image channels
    for (channel, color) in zip(channels, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

def plot_2d_histograms(image):
    # split the image into its respective channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # create a new figure for 2D color histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # plot 2D color histograms
    channels_pairs = [(green_channel, blue_channel), (green_channel, red_channel), (blue_channel, red_channel)]
    titles = ["G and B", "G and R", "B and R"]

    for ax, (channel1, channel2), title in zip(axes, channels_pairs, titles):
        hist = cv2.calcHist([channel1, channel2], [0, 1], None, [32, 32], [0, 256, 0, 256])
        p = ax.imshow(hist, interpolation="nearest")
        ax.set_title(f"2D Color Histogram for {title}")
        plt.colorbar(p, ax=ax)

def main():
    # Load the input image from disk
    image = cv2.imread("03_OpenCV-Histograms/01_image-histograms/image/beach.png")

    # Check if the image is loaded successfully
    if image is None:
        logger.error("Unable to load image.")
        return

    logger.info("Image loaded successfully.")

    # Plot histograms
    plot_histograms(image)
    plot_2d_histograms(image)

    # Display the original input image
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
