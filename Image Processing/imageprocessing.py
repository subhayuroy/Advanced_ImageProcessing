# -*- coding: utf-8 -*-
"""ImageProcessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p32u78fh3vzTuK3TgaF9Aj9NXw5ztAnC
"""

from scipy import ndimage
import numpy as np

"""###Opening and writing to image files"""

from scipy import misc
import imageio
f = misc.face()
imageio.imsave('face.png', f) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(f)
plt.show()

from scipy import misc
import imageio
face = misc.face()
imageio.imsave('face.png', face) # First we need to create the PNG file

face = imageio.imread('face.png')
type(face)

face.shape, face.dtype

face.tofile('face.raw') # Create raw file
face_from_raw = np.fromfile('face.raw', dtype=np.uint8)
face_from_raw.shape

face_from_raw.shape = (768, 1024, 3)

face_memmap = np.memmap('face.raw', dtype=np.uint8, shape=(768, 1024, 3))

for i in range(10):
    im = np.random.randint(0, 256, 10000).reshape((100, 100))
    imageio.imsave('random_%02d.png' % i, im)
from glob import glob
filelist = glob('random*.png')
filelist.sort()

"""###Displaying images"""

f = misc.face(gray=True)  # retrieve a grayscale image
import matplotlib.pyplot as plt
plt.imshow(f, cmap=plt.cm.gray)

plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)

plt.axis('off')

plt.contour(f, [50, 200])

"""###Basic manipulations"""

plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='bilinear')

plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='nearest')

face = misc.face(gray=True)
face[0, 40]

# Slicing
face[10:13, 20:23]

face[100:120] = 255

"""###Geometrical transformations"""

lx, ly = face.shape
X, Y = np.ogrid[0:lx, 0:ly]
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
# Masks
face[mask] = 0
# Fancy indexing
face[range(400), range(400)] = 25

face = misc.face(gray=True)
lx, ly = face.shape

# Cropping
crop_face = face[lx // 4: - lx // 4, ly // 4: - ly // 4]
plt.imshow(crop_face)

# up <-> down flip
flip_ud_face = np.flipud(face)
plt.imshow(flip_ud_face)

# rotation
rotate_face = ndimage.rotate(face, 45)
rotate_face_noreshape = ndimage.rotate(face, 45, reshape=False)
plt.imshow(rotate_face)

"""###Blurring/smoothing"""

from scipy import misc
face = misc.face(gray=True)
blurred_face = ndimage.gaussian_filter(face, sigma=3)
very_blurred = ndimage.gaussian_filter(face, sigma=5)
plt.imshow(very_blurred)

"""###Sharpening"""

from scipy import misc
face = misc.face(gray=True).astype(float)
blurred_f = ndimage.gaussian_filter(face, 3)

filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
plt.imshow(sharpened)

"""###Segmentation"""

n = 10
l = 256
im = np.zeros((l, l))
np.random.seed(1)
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = (im > im.mean()).astype(np.float)
mask += 0.1 * im
img = mask + 0.2*np.random.randn(*mask.shape)

hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

binary_img = img > 0.5

# Remove small white regions
open_img = ndimage.binary_opening(binary_img)
# Remove small black hole
close_img = ndimage.binary_closing(open_img)
plt.imshow(close_img)

"""###Edge Detection using Canny Edge Detector"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(16, 16))
img_gs = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gs.jpg', img_gs)
edges = cv2.Canny(img_gs, 100,200)
plt.subplot(121), plt.imshow(img_gs)
plt.title('Original Gray Scale Image')
plt.subplot(122), plt.imshow(edges)
plt.title('Edge Image')
plt.show()

"""###Measuring objects properties"""

n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
mask = im > im.mean()

label_im, nb_labels = ndimage.label(mask)
nb_labels

plt.imshow(label_im)