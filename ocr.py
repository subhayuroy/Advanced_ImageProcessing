import os
from PIL import Image
import pytesseract
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image that will be processed by OCR / tesseract")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="preprocessing method that is applied to the image")
args = vars(ap.parse_args())
