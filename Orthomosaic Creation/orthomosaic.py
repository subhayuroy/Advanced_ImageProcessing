# -*- coding: utf-8 -*-
"""
**Installing the appropiate version of OpenCV**
"""

!pip install opencv-contrib-python==3.4.2.16

"""##Working on "Gravel Quarry" dataset

**Importing packages**
"""

import cv2
import numpy as np

"""**Uploading 2 images**"""

img_ = cv2.imread('/content/IX-01-61737_0029_0053.JPG')
img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('/content/IX-01-61737_0029_0054.JPG')
img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

"""###Finding the key points and descriptors with SIFT"""

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

"""**Displaying the output on 1st image**"""

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
# cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
plt.imshow(cv2.drawKeypoints(img_,kp1,None))

"""###Finding the 'Matching' points between two images

####FLANN matcher
"""

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(des1,des2,k=2)

"""####BFMatcher matcher"""

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

"""Both will perform the same way and will give same output"""

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

"""**Output image with matches drawn**"""

draw_params = dict(matchColor = (255,0,0), # draw matches in red color
                   singlePointColor = None,
                   flags = 2)
img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
# cv2.imshow("original_image_drawMatches.jpg", img3)
plt.figure(figsize=(12,10))
plt.imshow(img3)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0],0:img.shape[1]] = img
# cv2.imshow("original_image_stitched.jpg", dst)
plt.figure(figsize=(14,10))
plt.imshow(dst)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def crop(image):
  y_nonzero, x_nonzero, _ = np.nonzero(image)
  return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

plt.figure(figsize=(12,8))
plt.imshow(crop(dst))

"""It seems the output will not be as expected. But, we can notice it is performing our main motive - 'Homography' and 'Matching between images'

##Working on "Dam Inspection" dataset

**Creating the Matcher class**

This deals with:
  1. Feature extraction
  2. Matching correspondences between images
"""

import cv2
import numpy as np

class matchers:
  def __init__(self):
    self.surf = cv2.xfeatures2d.SURF_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=0, trees=5) 
    search_params = dict(checks=50)
    self.flann = cv2.FlannBasedMatcher(index_params, search_params)

  def match(self, i1, i2, direction=None): 
    imageSet1 = self.getSURFFeatures(i1) 
    imageSet2 = self.getSURFFeatures(i2) 
    print("Direction : ", direction) 
    matches = self.flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
    good = []
    for i , (m, n) in enumerate(matches): 
      if m.distance < 0.7*n.distance:
        good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
      pointsCurrent = imageSet2['kp'] 
      pointsPrevious = imageSet1['kp']

      matchedPointsCurrent = np.float32( [pointsCurrent[i].pt for (_, i) in good])
      matchedPointsPrev = np.float32( [pointsPrevious[i].pt for (i,_ ) in good])

      H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4) 
      return H
    return None

  def getSURFFeatures(self, im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = self.surf.detectAndCompute(gray, None) 
    return {'kp':kp, 'des':des}

"""**Creating Panorama class**

This deals with:
  3. Compute Homography
  4. Wrapping ans Stitching
"""

import numpy as np 
import cv2
import sys
from matchers import Matchers
import time
import matplotlib.pyplot as plt

class Stitch:
  def __init__(self, args): 
    self.path = '/content/txtlists/files1.txt'
    fp = open(self.path, 'r')
    filenames = [each.rstrip('\r\n') for each in fp.readlines()] 
    print(filenames)
    self.images = [cv2.resize(cv2.imread(each),(427, 320)) for each in filenames] 
    self.count = len(self.images)
    self.left_list, self.right_list, self.center_im = [], [],None
    self.matcher_obj = Matchers() 
    self.prepare_lists()

  def prepare_lists(self):
    print("Number of images : %d"%self.count) 
    self.centerIdx = self.count/2
    print("Center index image : %d"%self.centerIdx) 
    self.center_im = self.images[int(self.centerIdx)] 
    for i in range(self.count):
      if(i<=self.centerIdx): 
        self.left_list.append(self.images[i])
      else:
        self.right_list.append(self.images[i]) 
        print("Image lists prepared")

  def leftshift(self):
    # self.left_list = reversed(self.left_list)
    a = self.left_list[0]
    for b in self.left_list[1:]:
      H = self.matcher_obj.match(a, b, direction='left') 
      print("Homography is : ", H)
      xh = np.linalg.inv(H)
      print("Inverse Homography :", xh)
      ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1])); 
      ds = ds/ds[-1]
      print("final ds=>", ds)
      f1 = np.dot(xh, np.array([0,0,1])) 
      f1 = f1/f1[-1]
      xh[0][-1] += abs(f1[0])
      xh[1][-1] += abs(f1[1])
      ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1])) 
      offsety = abs(int(f1[1]))
      offsetx = abs(int(f1[0]))
      dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety) 
      print("image dsize =>", dsize)
      tmp = cv2.warpPerspective(a, xh, dsize)
      # cv2.imshow("warped", tmp) # cv2.waitKey()
      tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b 
      a = tmp

    self.leftImage = tmp

  def rightshift(self):
    for each in self.right_list:
      H = self.matcher_obj.match(self.leftImage, each, 'right') 
      print("Homography :", H)
      txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
      txyz = txyz/txyz[-1]
      dsize = (int(txyz[0])+self.leftImage.shape[1],int(txyz[1])+self.leftImage.shape[0])
      tmp = cv2.warpPerspective(each, H, dsize) 
      plt.imshow(tmp)
      plt.show()
      #cv2.waitKey()
      # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
      tmp = self.mix_and_match(self.leftImage, tmp)
      print("tmp shape",tmp.shape)
      print("self.leftimage shape=", self.leftImage.shape) 
      self.leftImage = tmp
      # self.showImage('left')

  def mix_and_match(self, leftImage, warpedImage): 
    i1y, i1x = leftImage.shape[:2]
    i2y, i2x = warpedImage.shape[:2] 
    print(leftImage[-1,-1])

    t = time.time()
    black_l = np.where(leftImage == np.array([0,0,0])) 
    black_wi = np.where(warpedImage == np.array([0,0,0])) 
    print(time.time() - t)
    print(black_l[-1])

    for i in range(0, i1x): 
      for j in range(0, i1y):
        try:
          if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
            # print "BLACK"
            # instead of just putting it with black,
            # take average of all nearby values and avg it.
            warpedImage[j,i] = [0, 0, 0] 
          else:
            if(np.array_equal(warpedImage[j,i],[0,0,0])):
              # print "PIXEL"
              warpedImage[j,i] = leftImage[j,i] 
            else:
              if not np.array_equal(leftImage[j,i], [0,0,0]):
                bw, gw, rw = warpedImage[j,i] 
                bl,gl,rl = leftImage[j,i]
                # b = (bl+bw)/2
                # g = (gl+gw)/2 
                # r = (rl+rw)/2
                warpedImage[j, i] = [bl,gl,rl]
        except:
          pass
          # cv2.imshow("waRPED mix", warpedImage) 
          # cv2.waitKey()
    return warpedImage

  def trim_left(self): 
    pass

  def showImage(self, string=None): 
    if string == 'left':
      plt.imshow(self.leftImage)
      plt.show()
      # cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
    elif string == "right": 
      plt.imshow(self.rightImage) 
      plt.show()
      #cv2.waitKey()

if  __name__ 	== '__main__': 
  try:
    args = sys.argv[1] 
  except:
    args = "txtlists/files1.txt"
  finally:
    print("Parameters : ", args) 
  s = Stitch(args)
  s.leftshift()
  # s.showImage('left')
  s.rightshift() 
  print("Done")
  cv2.imwrite("image_mosaic1.jpg", s.leftImage) 
  print("Image written") 
  #cv2.destroyAllWindows()

"""**This seems to perform the actual work of Orthomosaic**

Stitches all the images given to it within a textfile

##Working on "Small Village" dataset
"""

from PIL import Image
import os
import cv2
import numpy as np
import sys
import time
from google.colab.patches import cv2_imshow

path = "/content/Village_Dataset/geotagged-images"
resize_ratio = 0.5 # where 0.5 is half size, 2 is double size
def resize_aspect_fit():
 dirs = os.listdir(path)
 for item in dirs:
   if item == '.JPG':
     continue
   if os.path.isfile(path+item):
     image = Image.open(path+item)
     file_path, extension = os.path.splitext(path+item)
     new_image_height = int(image.size[0] / (1/resize_ratio))
     new_image_length = int(image.size[1] / (1/resize_ratio))
     image = image.resize((new_image_height, new_image_length), Image.ANTIALIAS)
     image.save(file_path + "_small" + extension, 'JPEG', quality=90)
resize_aspect_fit()

class matchers:
  def __init__(self):
    self.surf = cv2.xfeatures2d.SURF_create()
    FLANN_INDEX_KDTREE = 5
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    self.flann = cv2.FlannBasedMatcher(index_params, search_params)

  def match(self, i1, i2, direction=None):
    imageSet1 = self.SURFfeature(i1)
    imageSet2 = self.SURFfeature(i2)
    matches = self.flann.knnMatch(imageSet2['des'],imageSet1['des'],k=2 )
    good = []
    for i , (m, n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
        good.append((m.trainIdx, m.queryIdx))
      if len(good) > 4:
        pointsCurrent = imageSet2['kp']
        pointsPrevious = imageSet1['kp']
        matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
        matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])
        #Computing Homography for the images
        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC,5.0)
        return H
      return None
  #Keypoints and Descriptors creation
  def SURFfeature(self, im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = self.surf.detectAndCompute(gray, None)
    return {'kp':kp, 'des':des}

class Stitch:
  def __init__(self, args):
    #Passing a txt file which contains image names
    self.path = '/content/files1.txt'
    filepath = open(self.path, 'r')
    filenames = [each.rstrip('\r\n') for each in filepath.readlines()]
    print(filenames)
    self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
    self.count = len(self.images)
    self.list1, self.list2, self.center_im = [], [],None
    self.matcher_obj = matchers()
    self.Images()

  def Images(self):
    self.centerIdx = self.count/2
    self.center_im = self.images[int(self.centerIdx)]
    for i in range(self.count):
      if(i<=self.centerIdx):
        self.list1.append(self.images[i])
      else:
        self.list2.append(self.images[i])
    print("Image lists prepared")

  def Stitchleft(self):
    a = self.list1[0]
    for b in self.list1[1:]:
      H = self.matcher_obj.match(a, b, 'left')
      xhomography = np.linalg.inv(H)
      dim = np.dot(xhomography, np.array([a.shape[1], a.shape[0], 1]));
      dim = dim/dim[-1]
      f1 = np.dot(xhomography, np.array([0,0,1]))
      f1 = f1/f1[-1]
      xh[0][-1] += abs(f1[0])
      xh[1][-1] += abs(f1[1])
      ds = np.dot(xhomography, np.array([a.shape[1], a.shape[0], 1]))
      offsety = abs(int(f1[1]))
      offsetx = abs(int(f1[0]))
      dimsize = (int(dim[0])+offsetx, int(dim[1]) + offsety)
      tmp = cv2.warpPerspective(a, xhomography, dsize)
      tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b 
      a = tmp
    self.warpImage = tmp

  def Stitchright(self):
    for each in self.right_list:
      H = self.matcher_obj.match(self.warpImage, each, 'right')
      txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
      txyz = txyz/txyz[-1]
      dimsize = (int(txyz[0])+self.warpImage.shape[1], int(txyz[1])+self.warpImage.shape[0])
      tmp = cv2.warpPerspective(each, H, dimsize)
      cv2_imshow(tmp)
      cv2.waitKey()
      tmp = self.StitchImages(self.warpImage, tmp)
    self.warpImage = tmp

  def StitchImages(self, warpImage, warped):
    i1y, i1x = warpImage.shape[:2]
    i2y, i2x = warped.shape[:2] 
    t = time.time()
    black_l = np.where(warpImage == np.array([0,0,0]))
    black_wi = np.where(warped == np.array([0,0,0]))
    for i in range(0, i1x):
      for j in range(0, i1y):
        try:
          if(np.array_equal(warpImage[j,i],np.array([0,0,0]))):
              warped[j,i] = [0, 0, 0]
          else:
            if(np.array_equal(warped[j,i],[0,0,0])):
              warped[j,i] = warpImage[j,i]
            else:
              if not np.array_equal(warpImage[j,i], [0,0,0]):
                bw, gw, rw = warped[j,i]
                bl,gl,rl = warpImage[j,i]
                warped[j, i] = [bl,gl,rl]
        except:
          pass
    return warped

  def trim_left(self):
    pass

  def showImage(self, string=None):
    if string == 'left':
      cv2_imshow("left image", self.warpImage)
    elif string == "right":
      cv2_imshow("right Image", self.warpright)
      cv2.waitKey()

if __name__ == '__main__':
  try:
    args = sys.argv[1]
  except:
    args = "content/files1.txt"
  finally:
    s = Stitch(args) 
    s.Stitchleft()
    s.Stitchright()
    cv2.imwrite("test12.jpg", s.warpImage)
    cv2.destroyAllWindows()

"""***Because of change in datasets and not many Common/Matching points are available. This Algorithm fails in this Dataset.***

**Importing necessary packages**
"""

import cv2
import numpy as np
import matplotlib.pyplot as pit 
import pandas as pd
from random import randrange 
import argparse
import glob

"""Giving/Supplying the required images"""

img1 = cv2.imread(r'/content/IMG_7747.JPG') 
img2 = cv2.imread(r'/content/IMG_7748.JPG') 
img3 = cv2.imread(r'/content/IMG_7749.JPG')

"""Re-sizing those images"""

img1 = cv2.resize(img1, (0,0), fx=1, fy=1) 
img2 = cv2.resize(img2, (0,0), fx=1, fy=1) 
img3 = cv2.resize(img3, (0,0), fx=1, fy=1)

def images_gray(img1,img2,img3):
  imglgray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) 
  img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) 
  img3gray = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY) 
  return imglgray,img2gray,img3gray

img1gray,img2gray,img3gray = images_gray(img1,img2,img3)

"""*Changing the color to GrayScale mode*

**Displaying all the 3 images**
"""

figure, ax = plt.subplots(1, 3, figsize=(18, 10))
ax[0].imshow(img1gray, cmap='gray') 
ax[1].imshow(img2gray, cmap='gray') 
ax[2].imshow(img3gray, cmap='gray')

"""**Using the ORB Algorithm**"""

orb = cv2.ORB_create()
#detect keypoints and extract
kp1,des1 = orb.detectAndCompute(img1, None) 
kp2,des2 = orb.detectAndCompute(img2, None) 
kp3,des3 = orb.detectAndCompute(img3, None)

keypoints_with_size1 = np.copy(img1)
keypoints_with_size2 = np.copy(img2)
keypoints_with_size3 = np.copy(img3)

"""Drawing the keypoints on one of the images given(here, 1st image)"""

cv2.drawKeypoints(img1, kp1, keypoints_with_size1, color=(0,255,0))

"""**Displaying the keypoints**"""

cv2.drawKeypoints(img1, kp1, keypoints_with_size1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.drawKeypoints(img2, kp2, keypoints_with_size2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.drawKeypoints(img3, kp3, keypoints_with_size3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
plt.figure(figsize=(20,10))
plt.subplot(1, 3, 1)
plt.title("Image 1") 
plt.imshow(keypoints_with_size1, cmap='gray') 
plt.subplot(1, 3, 2)
plt.title("Image 2") 
plt.imshow(keypoints_with_size2, cmap='gray') 
plt.subplot(1, 3, 3)
plt.title("Image 3") 
plt.imshow(keypoints_with_size3, cmap='gray')

print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape)) 
print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape)) 
print("keypoints: {}, descriptors: {}".format(len(kp3), des3.shape))

"""**Using the SIFT Algorithm**"""

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None) 
kp2, des2 = sift.detectAndCompute(img2,None) 
kp3, des3 = sift.detectAndCompute(img3,None)

"""***Brute-Force (BF) Matcher***"""

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)
print('Number of raw matches: %d.' % len(matches))

"""Applying ratio test"""

good = [ ]
for m, n in matches:
  if m.distance < 0.7*n.distance : 
    good.append([m])
matches = np.asarray(good)

img4 =  cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2) 
plt.figure(figsize=(17,10))
plt.title("ORB Feature matching") 
plt.imshow(img4)

"""Finding the Homography"""

if len(matches[:,0]) >= 4:
  src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2) 
  dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2) 
  H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
  raise AssertionError("Can't find enough keypoints.")

"""Warping these images together"""

dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0])) 
dst[0:img2.shape[0], 0:img2.shape[1]] = img2 
plt.figure(figsize=[15,10]),plt.title('Warped Image')
plt.imshow(dst) 
plt.show()

def crop(image):
  y_nonzero, x_nonzero, _ = np.nonzero(image)
  return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

plt.figure(figsize=(15,10))
plt.imshow(crop(dst), cmap='gray')
output = crop(dst)

"""**After cropping the Warped images**

Detecting keypoints and descriptors for the 3rd image
"""

sift = cv2.xfeatures2d.SIFT_create()

kp4, des4 = sift.detectAndCompute(output,None)

keypoints_with_size4 = np.copy(output)
cv2.drawKeypoints(output, kp4, keypoints_with_size4, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

"""####Comparing with the 3rd image"""

plt.figure(figsize=(15,10)) 
plt.subplot(1, 3, 1)
plt.title("Image 1") 
plt.imshow(keypoints_with_size4, cmap='gray') 
plt.subplot(1, 3, 2)
plt.title("Image 2") 
plt.imshow(keypoints_with_size3, cmap='gray')

bf = cv2.BFMatcher()
matches = bf.knnMatch(des3,des4, k=2)
print("keypoints: {}, descriptors: {}".format(len(kp3), des3.shape)) 
print("keypoints: {}, descrlptors: {}".format(len(kp4), des4.shape))

good = []
for m,n in matches:
  if m.distance < 0.7*n.distance: 
    good.append([m])
matches = np.asarray(good)

"""**SIFT Feature matching**"""

img5 = cv2.drawMatchesKnn(img3,kp3,output,kp4,matches,None,flags=2) 
plt.figure(figsize=(15,10))
plt.title("Sift Feature matching") 
plt.imshow(img5)

if len(matches[:,0]) >= 4:
  src = np.float32([ kp3[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2) 
  dst = np.float32([ kp4[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2) 
  H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
  raise AssertionError("Can’t find enough keypoints.")

dst = cv2.warpPerspective(img3,H,(output.shape[1] + img3.shape[1], output.shape[0])) 
dst[0:output.shape[0], 0:output.shape[1]] = output 
plt.figure(figsize=[15,10]),plt.title('Warped Image')
plt.imshow(dst) 
plt.show()

"""Stitched/Warped image is generated, only croping is left"""

plt.figure(figsize=(15,10))
plt.imshow(crop(dst), cmap='gray')
output2 = crop(dst)

"""***The final image after stitching 3 images together to form an Orthomosaic**

This was another successful result.

###Trying another method of stitching in this dataset
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv.imread('/content/IMG_7755.JPG', cv.IMREAD_GRAYSCALE) 
img_2 = cv.imread('/content/IMG_7754.JPG', cv.IMREAD_GRAYSCALE)

plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.title('Image 1')
plt.imshow(img_1)
plt.subplot(1,2,2)
plt.imshow(img_2)
plt.title('Image 2')

akaze = cv.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(img_1, None)
kp2, des2 = akaze.detectAndCompute(img_2, None)

img1 = cv.imread('IMG_7755.JPG')
img2 = cv.imread('IMG_7754.JPG')

img1_key=cv.drawKeypoints(img_1, kp1,img1,(0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('keypoints.jpg', img1_key)
plt.figure(figsize=(10,10))
plt.imshow(img1_key)
plt.show()

img2_key=cv.drawKeypoints(img_2, kp2,img2,(0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('keypoints2.jpg', img2_key)
plt.figure(figsize=(10,10))
plt.imshow(img2_key)
plt.show()

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m,n in matches:
 if m.distance < 0.75*n.distance:
   good_matches.append([m])

img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('matches.jpg', img4)
plt.figure(figsize=(15,15))
plt.imshow(img4)
plt.show()

ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
# Compute homography
H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)

h=img_1.shape[1]+img_2.shape[1] 
w=img_1.shape[0]
warped_image = cv.warpPerspective(img_1, H, (h,w))
 
cv.imwrite('warped.jpg', warped_image)
plt.figure(figsize=(15,15))
plt.imshow(warped_image)
plt.show()

def crop(img):
 y,x=np.nonzero(img)
 return img[np.min(y):np.max(y),np.min(x):np.max(x)]

cv.imwrite('cropped.jpg', crop(warped_image))
plt.figure(figsize=(15,15))
plt.imshow(crop(warped_image))
plt.show()

img_3 = cv.imread('/content/IMG_7753.JPG', cv.IMREAD_GRAYSCALE) 
img3 = cv.imread('/content/IMG_7753.JPG')
img_4=cv.imread('/content/cropped.jpg')

plt.imshow(img_3)

akaze = cv.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img_3, None)
kp2, des2 = akaze.detectAndCompute(img_4, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches1 = []
for m,n in matches:
 if m.distance < 0.75*n.distance:
   good_matches1.append([m])

img5 = cv.drawMatchesKnn(img_3,kp1,img_4,kp2,good_matches1,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('matches1.jpg', img5)
plt.figure(figsize=(15,15))
plt.imshow(img5)
plt.show()

ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches1]).reshape(-1,1,2)
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches1]).reshape(-1,1,2)

H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)

h=img_3.shape[1]+img_4.shape[1] 
w=img_3.shape[0]
warped_image1 = cv.warpPerspective(img_3, H, (h,w))
 
cv.imwrite('warped1.jpg', warped_image1)
plt.figure(figsize=(15,15))
plt.imshow(warped_image1)
plt.show()

cv.imwrite('cropped1.jpg', crop(warped_image1))
plt.figure(figsize=(15,15))
plt.imshow(crop(warped_image1))
plt.show()

import imutils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

img_dir = r"/content/Village_Dataset/geotagged-images" 
data_path = os.path.join(img_dir,'*G') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
 img = cv.imread(f1) 
 data.append(img)

plt.figure(figsize=[15,15])
for i in range(6):
 plt.subplot(3,3,i+1)
 plt.imshow(data[i])
plt.show()

print("[INFO] stitching images...")
stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(data)

if status == 0:
 cv.imwrite("stiched_img.jpg", stitched)
 plt.figure(figsize=[15,15])
 plt.imshow(stitched)
 plt.show()
else:
 print("[INFO] image stitching failed ({})".format(status))

plt.figure(figsize=[15,15])
for i in range(6):
 plt.subplot(3,3,i+1)
 plt.imshow(data[i])
plt.show()

print("[INFO] stitching images...")
stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(data)

if status == 0:
 cv.imwrite("stiched_img.jpg", stitched)
 plt.figure(figsize=[15,15])
 plt.imshow(stitched)
 plt.show()
else:
 print("[INFO] image stitching failed ({})".format(status))

plt.figure(figsize=[15,15])
for i in range(6):
 plt.subplot(3,3,i+1)
 plt.imshow(data[i])
plt.show()

print("[INFO] stitching images...")
stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(data)

if status == 0:
 cv.imwrite("stiched_img.jpg", stitched)
 plt.figure(figsize=[15,15])
 plt.imshow(stitched)
 plt.show()
else:
 print("[INFO] image stitching failed ({})".format(status))

"""##After many experiments, this is the final working of Orthomosaic. 
This stitches **6 images** at once.
"""
