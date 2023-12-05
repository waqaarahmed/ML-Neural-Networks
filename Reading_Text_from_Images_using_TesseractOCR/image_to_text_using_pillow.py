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


import PIL.Image as pim
import pytesseract

#configuring the page segmentation mode and the OCR engine mode
config = r"--psm 3 --oem 3"

#getting text from image using image_to_string() function
text = pytesseract.image_to_string(pim.open("sign.webp"), config=config)
print(text)