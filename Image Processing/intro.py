import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

img = cv2.imread('burano.jpg')  #Import the image
#plt.imshow(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert the image into RGB
#plt.imshow(img_rgb)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Convert the image to Grayscale
#plt.imshow(img_gray, cmap='gray')

fig, axs = plt.subplots(nrows=1, bnols=3, figsize=(20,20))

for i in range(3):
  ax = axs[i]
  ax.imshow(img_rgb[:,:,i], cmap='gray')
plt.show()
