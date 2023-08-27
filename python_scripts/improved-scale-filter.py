# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 07:35:53 2022

@author: aidan
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import color, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import argparse
import imutils



img = cv2.imread("C:/Users/aidan/Downloads/Particles/Particles/L2_00a6b5e9806a8b072b98fdeacb3f45b5.jpg")
# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create binary mask based on threshold of 250
ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

#%%
# =============================================================================
# cv2.imshow("final final", binary)
# cv2.waitKey(0)
# =============================================================================


# create tall thin kernel
verticle_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, 
    (1, 13)
)
# create short long kernel
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, 
    (13, 1)
)
# detect vertical lines in the image
img_v = cv2.erode(binary, verticle_kernel, iterations = 3)
img_v = cv2.dilate(img_v, verticle_kernel, iterations = 3)


# detect horizontal lines in the image
img_h = cv2.erode(binary, horizontal_kernel, iterations = 3)
img_h = cv2.dilate(img_h, horizontal_kernel, iterations = 3)




kernel=np.ones((3,3), np.uint8)
# add the vertically eroded and horizontally eroded images
img_add = cv2.addWeighted(img_v, 0.5, img_h, 0.5, 0.0)
# perform final erosion and binarization
img_final = cv2.erode(~img_add, kernel, iterations = 1)
ret, img_final = cv2.threshold(
    img_final, 
    128, 
    255, 
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)

# =============================================================================
# cv2.imshow("added",img_add)
# cv2.imshow("final",img_final)
# cv2.waitKey(0)
# =============================================================================


banner = np.argwhere(img_final == 0)
# coordinates of the top left corner
banner_x1, banner_y1 = banner[0, 1], banner[0, 0]
# coordinates of the bottom right corner
banner_x2, banner_y2 = banner[-1, 1], banner[-1, 0]
# calculate width and height of banner
banner_width = banner_x2 - banner_x1
banner_height = banner_y2 - banner_y1

# finding the reflection below the banner
bot_reflect = img[
    banner_y2:banner_y2 + banner_height // 2, 
    banner_x1:banner_x2, 
    :
]
bot_reflect = np.flipud(bot_reflect)
# finding the reflection above the banner
top_reflect = img[
    banner_y1 - (banner_height - len(bot_reflect)):banner_y1,    
    banner_x1:banner_x2, 
    :
]
top_reflect = np.flipud(top_reflect)

reflect_pad = np.concatenate((top_reflect, bot_reflect), axis = 0)
imgcopy = img.copy()
imgcopy[banner_y1:banner_y2, banner_x1:banner_x2] = reflect_pad


cv2.imshow("final final", imgcopy)
cv2.waitKey(0)

#%%


photo_path = ("C:/Users/aidan/Documents/Nanoparticle_Detection/Data/Images/")
file_path = ("C:/Users/aidan/Documents/Nanoparticle_Detection/Data/")

img = imgcopy
# reads the image in grayscale form.

# remove noise using a mean filter
shifted = cv2.pyrMeanShiftFiltering(imgcopy, 21, 51)
# threshold the image 
graycopy = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(
    graycopy, 
    0, 
    255, 
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)

cv2.imshow("img", shifted)
cv2.imshow("img1", thresh)
cv2.waitKey(0)



kernel=np.ones((3,3), np.uint8)
# setting a kernel for the eroding and dilating functions


dilated = cv2.erode(thresh, kernel, iterations = 1)
eroded =cv2.dilate(dilated, kernel, iterations = 1)
# erodes and dilates the image so as to remove noise and clean the particles

cv2.imshow("eroded 1",dilated)
cv2.imshow("dilated 1", eroded)
cv2.waitKey(0)

#dilated = clear_border(dilated) #Remove nanoparticles touching the edges
#eroded = clear_border(eroded)

# =============================================================================
# # apply adistance transform and find the local maxima
# dt = cv2.distanceTransform(thresh, 2, 3)
# localmax = peak_local_max(
#     dt, 
#     indices = False, 
#     min_distance = 20, 
#     labels = thresh
# )
#     
# # apply the watershed algorithm
# markers = ndimage.label(localmax, structure=np.ones((3, 3)))[0]
# labels = watershed(-dt, markers, mask = thresh)
# =============================================================================

mask = eroded == 255 # Creating a mask which only includes white pixels from the dilated image






s = [[1,1,1], [1,1,1], [1,1,1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)
# Using ndimage to label the all particles that showed up in the mask

img2 = color.label2rgb(labeled_mask, bg_label = 0)
# coloring these labeled sections to better portray the labeling

dilated = np.uint8(eroded)
# changing the format of the dilated image so we can save it



save1 = (photo_path + '0_erode.png')
status1 = cv2.imwrite(save1, eroded)
print("Eroded Image written to file-system : ",status1)

save3 = (photo_path + '0_thresh.png')
status3 = cv2.imwrite(save3, thresh)
print("Thresholded Image written to file-system : ",status3)

#save = (photo_path + '7a.png')
#status = cv2.imwrite(save, dilated)
#print("Dilated Image written to file-system : ",status)

img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
save2 = (photo_path + '0b.png')
status2 = cv2.imwrite(save2, img2)
print("Colored Image written to file-system : ",status2)
