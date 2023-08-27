# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 03:27:47 2022

@author: aidan
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from PIL import Image

image1 = cv2.imread("C:/Users/aidan/Documents/Nanoparticle_Detection/Images/7.jpg")
img = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

photo_path = ("C:/Users/aidan/Documents/Nanoparticle_Detection/Data/Images/")
file_path = ("C:/Users/aidan/Documents/Nanoparticle_Detection/Data/")

# Thresholding the Image using the OTSU algorithim
ret1, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



#%%

# Removing the small holes within the image

kernel = np.ones((3,3),np.uint8)
holes = cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

img0 = Image.fromarray(img_thresh)
img0.save('hello.png')
img0.show()

# Removing any grains that touch the border
from skimage.segmentation import clear_border
holes = clear_border(holes)
 
#%%
# Identifying what is for sure background so that watershed knowns it is not particle
background = cv2.dilate(holes,kernel,iterations=2)


# https://www.tutorialspoint.com/opencv/opencv_distance_transformation.html
# Finding what is definitely a particle by applying a distance transform and then thresholding
# (distance transforming gives pixels more intensity if they are further away 
# from the nearest 0 value(background))
distance = cv2.distanceTransform(holes,cv2.DIST_L2,3)

#%%
# convert values to 0 - 255 int8 format
formatted = (distance * 255 / np.max(distance)).astype('uint8')
img1 = Image.fromarray(formatted)
img1.show()

# Here you can clearly see that the distance transform is highlighting the pixels that are the 
# furthest away from any black edges, marking them as the "center"

#%%

# Thresholding the distance transform by about 20% of its max value
ret2, foreground = cv2.threshold(distance,0.2*distance.max(),255,0)


# 20% of the max value seperates the cells well, however, this variable is free to change
# to get better results on a different image. High percentages like 50% will not recognize some of the boundaries.
# 0.2* max value seems to separate the cells well.
foreground = np.uint8(foreground)

# Here we create an unknown region that is the background minus the foreground
unknown_region = cv2.subtract(background,foreground)


img2 = Image.fromarray(background)
img3 = Image.fromarray(foreground)
img4 = Image.fromarray(unknown_region)
img2.show()
img3.show()
img4.show()

#%%

# Here we are creating markers out of ConnectedComponents. We will give unknown regions a value of 0, and 
# our foreground and background will be labeled with positive numbers.
ret3, markers = cv2.connectedComponents(foreground)
markers = markers+10

# We mark the unknown region with a value of 0.
markers[unknown_region==255] = 0
# This plot clearly shows the unknown region, followed by the foreground and background.
plt.imshow(markers, cmap='jet')

# Now we use watershedding to mark the boundary layers that water "spills" down.
markers = cv2.watershed(image1,markers)
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1


# OpenCV assigns the boundaries a value of -1 so we will set that to red to be able to see it.

image1[markers == -1] = [255,0,0]  



image2 = color.label2rgb(markers, bg_label=0)

img5 = Image.fromarray(image1)
img5.show()

cv2.imshow("colored image",image2)
cv2.waitKey(0)

# Here we can clearly see how we cleaned the image using the markers and the 
# watershedding

#%%

# The last step is to measure the region properties, explained in more detail in
# the midterm project report.

clusters = measure.regionprops(markers, intensity_image=img)
#%%
propList = ['Area',
            'equivalent_diameter',
            'orientation',
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity',
            'eccentricity',
            'centroid'
            ] 
labelList = ['Label',
            'Area',
            'equivalent_diameter', 
            'orientation',
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity',
            'eccentricity',
            'y.cord',
            'x.cord'
            ]    

dataPath = ("C:/Users/aidan/Documents/Nanoparticle_Detection/Data/")
file = ("7_data.csv")
pathFile = dataPath + file

output_file = open(pathFile, 'w')
output_file.write(",".join(labelList) + '\n') #join strings in array by commas


# Here we have the option to input a scaling mechanic in our code. If we want all our measurements to be in realworld units such as meters, we can do that here.
for cluster_props in clusters:
    #output cluster properties to the excel file
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = cluster_props[prop] # *(pixels to nanometer)**2   
        elif(prop == 'orientation'): 
            to_print = cluster_props[prop]*57.2958  # Convert to degrees from radians
        elif(prop.find('Axis') > 0):
            to_print = cluster_props[prop] # * (pixels to nanometer)
        else: 
            to_print = cluster_props[prop]
        output_file.write(',' + str(to_print))
    output_file.write('\n') 
output_file.close() 
