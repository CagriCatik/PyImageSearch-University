# import the necessary packages
from skimage.exposure import is_low_contrast
import numpy as np
import imutils
import cv2
import logging

# specify the input video file path and threshold for low contrast
input_video = "03_OpenCV-Histograms/06_detect-low-contrast/example_video.mp4"
thresh = 0.5

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# grab a pointer to the input video stream
logger.info("Accessing video stream...")
vs = cv2.VideoCapture(input_video)

# initialize flag to indicate whether stream is paused
paused = False

# loop over frames from the video stream
while True:
    # read a frame from the video stream
    if not paused:
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed then we've reached the end of
        # the video stream so exit the script
        if not grabbed:
            logger.info("No frame read from stream - exiting")
            break

        # resize the frame, convert it to grayscale, blur it, and then
        # perform edge detection
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        # initialize the text and color to indicate that the current
        # frame is *not* low contrast
        text = "Low contrast: No"
        color = (0, 255, 0)

        # check to see if the frame is low contrast, and if so, update
        # the text and color
        if is_low_contrast(gray, fraction_threshold=thresh):
            text = "Low contrast: Yes"
            color = (0, 0, 255)

        else:
            # find contours in the edge map and find the largest one,
            # which we'll assume is the outline of our color correction
            # card
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # draw the largest contour on the frame
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        # draw the text on the output frame
        cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # stack the output frame and edge map next to each other
        output = np.hstack([frame, np.dstack([edged] * 3)])

        # show the output to our screen
        cv2.imshow("Output", output)

    # wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        logger.info("User pressed 'q', exiting...")
        break
    elif key == ord(" "):  # if space bar is pressed, toggle pause
        paused = not paused

# release the pointer to the input video stream and close the window
vs.release()
cv2.destroyAllWindows()
