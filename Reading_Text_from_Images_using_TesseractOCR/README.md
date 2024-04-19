# Reading Text from Images using Tesseract OCR

This Python script demonstrates how to extract text from images using the Tesseract OCR (Optical Character Recognition) engine.
It provides examples using both OpenCV and Pillow libraries for image processing.

## Using OpenCV
```
import cv2 as cv
import pytesseract as pyt
from pytesseract import Output

# Configuring the page segmentation mode and the OCR engine mode
config = r"--psm 3 --oem 3"

img = cv.imread("v1.png")

# Extracting text data from the image
data = pyt.image_to_data(img, config=config, output_type=Output.DICT)

# Processing detected text boxes
for i in range(len(data['text'])):
    if float(data['conf'][i]) > 20:
        (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
        img = cv.putText(img, data['text'][i], (x, y+height+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

cv.imshow("img", img)
cv.waitKey(0)
```

### Using Pillow
```
import PIL.Image as pim
import pytesseract

# Configuring the page segmentation mode and the OCR engine mode
config = r"--psm 3 --oem 3"

# Getting text from image using image_to_string() function
text = pytesseract.image_to_string(pim.open("sign.webp"), config=config)
print(text)
```

#### Note

- The script utilizes Tesseract OCR for text extraction from images.
- You can configure the page segmentation mode and OCR engine mode according to your requirements.
- Adjust the confidence threshold (conf) as needed to filter out low-confidence text detections.
- Feel free to customize the configurations and integrate these examples into your projects for text extraction from images.

##### Requirements

- Python 3.x
- Libraries: `opencv-python`, `pytesseract`, `Pillow`

You can install the required libraries using pip:

```bash
pip install opencv-python pytesseract Pillow
