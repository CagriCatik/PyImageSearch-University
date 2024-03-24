import cv2
import imutils
import numpy as np
import sys
import logging
from imutils.perspective import four_point_transform
from skimage import exposure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters()

    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,
        arucoDict, parameters=arucoParams)

    if not corners:  # Check if no markers were detected
        logger.error("No markers detected in the image.")
        return None

    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()

        # extract the coordinates of the markers
        markers = {}
        for marker_id in [923, 1001, 241, 1007]:
            i = np.squeeze(np.where(ids == marker_id))
            markers[marker_id] = np.squeeze(corners[i])[0]

    except Exception as e:
        logger.error(f"Error extracting marker coordinates: {e}")
        return None

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, birds-eye-view of the color
    # matching card
    cardCoords = np.array([markers[923], markers[1001], markers[241], markers[1007]])
    card = four_point_transform(image, cardCoords)

    # return the color matching card to the calling function
    return card



def main(reference_path, input_paths):
    # load the reference image from disk
    logger.info("Loading reference image...")
    ref = cv2.imread(reference_path)

    # resize the reference image
    ref = imutils.resize(ref, width=600)

    # display the reference image
    cv2.imshow("Reference", ref)

    # find the color matching card in the reference image
    logger.info("Finding color matching card in the reference image...")
    refCard = find_color_card(ref)

    # if the color matching card is not found in the reference image, exit
    if refCard is None:
        logger.error("Could not find color matching card in the reference image")
        sys.exit(1)

    # show the color matching card in the reference image
    cv2.imshow("Reference Color Card", refCard)

    # iterate over each input image
    for input_path in input_paths:
        # load the input image from disk
        logger.info(f"Loading input image: {input_path}...")
        image = cv2.imread(input_path)

        # resize the input image
        image = imutils.resize(image, width=600)

        # display the input image
        cv2.imshow("Input", image)

        # find the color matching card in the input image
        logger.info("Finding color matching card in the input image...")
        imageCard = find_color_card(image)

        # if the color matching card is not found in the input image, skip to the next image
        if imageCard is None:
            logger.error("Could not find color matching card in the input image")
            continue

        # show the color matching card in the input image
        cv2.imshow("Input Color Card", imageCard)

        # apply histogram matching from the color matching card in the reference image to the color matching card in the input image
        logger.info("Matching images...")
        corrected_imageCard = exposure.match_histograms(imageCard, refCard, multichannel=True)

        # show the input color matching card after histogram matching
        cv2.imshow("Corrected Input Color Card", corrected_imageCard)

        cv2.waitKey(0)

if __name__ == "__main__":
    # Provide paths to the reference and input images
    reference_path = r"03_OpenCV-Histograms\05_color-correction\reference.jpg"
    input_paths = ["examples/01.jpg", "examples/02.jpg", "examples/03.jpg"]
    main(reference_path, input_paths)
