import os
from PIL import Image
import pytesseract
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image that will be processed by OCR / tesseract")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="preprocessing method that is applied to the image")
args = vars(ap.parse_args())


# The image is loaded into memory – Python kernel
image = cv2.imread(args["image"])
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# preprocess the image
if args["preprocess"] == "thresh": gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# blur the image to remove noise
elif args["preprocess"] == "blur": gray = cv2.medianBlur(gray, 3)

# write the new grayscale image to disk 
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


# load the image as a PIL/Pillow image, apply OCR 
text = pytesseract.image_to_string(Image.open(filename))
print(text)
# show the output image
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
