import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('burano.jpg')
#plt.imshow(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert the image into RGB
#plt.imshow(img_rgb)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Convert the image to Grayscale
#plt.imshow(img_gray, cmap='gray')

fig, axs = plt.subplots(nrows=1, bnols=3, figsize=(20,20))  #Plot the three channels of the image
for i in range(3):
  ax = axs[i]
  ax.imshow(img_rgb[:,:,i], cmap='gray')
#plt.show()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

fig, (axs1, axs2) = plt.subplots(nrows=1, ncols=2, figsize=(20,20))
axs1.imshow(img_hsv)
axs2.imshow(img_hls)
plt.show()
