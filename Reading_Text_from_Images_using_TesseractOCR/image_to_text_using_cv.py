"""
Page segmentation modes:

O Orientation and script detection (OSD) only

1 Automatic page segmentation with OSD. ‘

2 Automatic page segmentation, but no OSD, or OCR.

3 Fully automatic page segmentation, but no OSD. (Default)

4 Assume a single column of text of variable sizes.

5 Assume a single uniform block of vertically aligned text.

6 Assume a single uniform block of textJ

7 Treat the image as a single text line.

8 Treat the image as a single word.

9 Treat the image as a single word in a circle.

10 Treat the image as a single character.

11 Sparse text. Find as much text as possible in no particular order.

12 Sparse text with OSD.

13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.
"""

"""
OCR Engine Mode
0 Legacy engine only.
1 Neural nets LSTM engine only.
2 Legacy + LSTM engines.
3 Default, based on what is available."""

import cv2 as cv
import pytesseract as pyt
from pytesseract import Output

#configuring the page segmentation mode and the OCR engine mode
config = r"--psm 3 --oem 3"

img = cv.imread("v1.png")
height, width, _ = img.shape

data = pyt.image_to_data(img, config=config, output_type=Output.DICT)
boxes = len(data['text'])
for i in range(boxes):
    if float(data['conf'][i]) > 20:
        (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
        img = cv.putText(img, data['text'][i], (x, y+height+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

cv.imshow("img", img)
cv.waitKey(0)

