# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:14:53 2022

@author: aidan
"""

import cv2
# Imports OpenCV for Image Processing
import numpy as np
from scipy import ndimage
# Imports ndimage from scipy for image labeling
from skimage import color, measure
# Imports Coloring and Measuring features from Skimage to output useful information


img = cv2.imread("C:/Users/aidan/Downloads/283.jpg")
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
img_final = cv2.erode(~img_add, kernel, iterations = 3)
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


# =============================================================================
# # finding the reflection below the banner
# bot_reflect = img[
#     banner_y2:banner_y2 + banner_height // 2, 
#     banner_x1:banner_x2, 
#     :
# ]
# bot_reflect = np.flipud(bot_reflect)
# # finding the reflection above the banner
# top_reflect = img[
#     banner_y1 - (banner_height - len(bot_reflect)):banner_y1,    
#     banner_x1:banner_x2, 
#     :
# ]
# top_reflect = np.flipud(top_reflect)
# 
# reflect_pad = np.concatenate((top_reflect, bot_reflect), axis = 0)
# imgcopy = img.copy()
# imgcopy[banner_y1:banner_y2, banner_x1:banner_x2] = reflect_pad
# =============================================================================


crop1 = gray[0:banner_y1, 0:1024]
crop2 = gray[banner_y2:768, 0:1024]
img10 = np.concatenate((crop1, crop2), axis=0)


#cv2.imshow("imgcopy", imgcopy)
#cv2.imshow("img", img)
#cv2.waitKey(0)

img10 = np.uint8(img10)
#%%
# reads the image in grayscale form.

# plt.hist(img.flat, bins=100, range=(0,255))
ret, thresh = cv2.threshold(img10, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# performs thresholding onto the image. THRESH_OTSU takes the above histogram and uses an algorithim developed by Nobuyuki Otsu to divide the pixels into two groups,
# foreground and backgroung, where it sets a threshold between these two groups.



#%%
kernel=np.ones((3,3), np.uint8)
# setting a kernel for the eroding and dilating functions

eroded =cv2.erode(thresh, kernel, iterations = 5)
dilated = cv2.dilate(eroded, kernel, iterations = 5)
# erodes and dilates the image so as to remove noise and clean the particles

#%%

mask = dilated == 255 # Creating a mask which only includes white pixels from the dilated image



s = [[1,1,1], [1,1,1], [1,1,1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)
# Using ndimage to label the all particles that showed up in the mask

img2 = color.label2rgb(labeled_mask, bg_label = 0)
# coloring these labeled sections to better portray the labeling

cv2.imshow("final", img2)
cv2.imshow("eroded", eroded)
cv2.imshow("dilated", dilated)
cv2.waitKey(0)

dilated = np.uint8(dilated)
# changing the format of the dilated image so we can save it


#%%

dilated = cv2.dilate(eroded, kernel, iterations = 8)
img3 = cv2.subtract(img10, dilated)


cv2.imshow("img3", img3)
cv2.waitKey(0)


#%%

image_path = "C:/Users/aidan/Documents/Nanoparticle_Detection/Images/Particles/"
photo_path = "C:/Users/aidan/Documents/Nanoparticle_Detection/Data/Images/"



sav = (photo_path + '0.png')
statu = cv2.imwrite(sav, img10)
print("Eroded Image written to file-system : ",statu)

save1 = (photo_path + '0a.png')
status1 = cv2.imwrite(save1, thresh)
print("Eroded Image written to file-system : ",status1)

save3 = (photo_path + '0b.png')
status3 = cv2.imwrite(save3, dilated)
print("Thresholded Image written to file-system : ",status3)

save = (photo_path + '0c.png')
status = cv2.imwrite(save, img10)
print("Dilated Image written to file-system : ",status)

save2 = (photo_path + '0d.png')
status2 = cv2.imwrite(save2, img3)
print("Colored Image written to file-system : ",status2)


#%%


img3[img3==0] = np.mean(img3)
cv2.imshow("img3",img3)
cv2.waitKey(0)

#%%
save4 = (photo_path + '0e.png')
status4 = cv2.imwrite(save4, img3)
print("Colored Image written to file-system : ",status4)

#%%
# plt.hist(img.flat, bins=100, range=(0,255))
ret, threshed = cv2.threshold(
    img3, 
    0, 
    255, 
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
# performs thresholding onto the image. THRESH_OTSU takes the above histogram and uses an algorithim developed by Nobuyuki Otsu to divide the pixels into two groups,
# foreground and backgroung, where it sets a threshold between these two groups.


kernel=np.ones((3,3), np.uint8)
# setting a kernel for the eroding and dilating functions

wane =cv2.erode(threshed, kernel, iterations = 2)
wax = cv2.dilate(wane, kernel, iterations = 2)
# erodes and dilates the image so as to remove noise and clean the particles
#cv2.imshow("eroded",wane)
#cv2.imshow("dilated",wax)

cv2.imshow("img",img3)
cv2.imshow("threshed",threshed)
cv2.imshow("wane",wane)
cv2.imshow("wax",wax)
cv2.waitKey(0)

#%%
mask2 = wax == 255 # Creating a mask which only includes white pixels from the dilated image


s2 = [[1,1,1], [1,1,1], [1,1,1]]
labeling, num_labels2 = ndimage.label(mask2, structure=s)
# Using ndimage to label the all particles that showed up in the mask

img8 = color.label2rgb(labeling, bg_label = 0)
# coloring these labeled sections to better portray the labeling


cv2.imshow("thresh", threshed)
cv2.imshow("final", img8)
cv2.waitKey(0)


#%%

final_image = cv2.add(img2,img8)



cv2.imshow("img1",img2)
cv2.imshow("img2",img8)
cv2.imshow("final image", final_image)
cv2.waitKey(0)

save5 = (photo_path + '0f.png')
status5 = cv2.imwrite(save5, final_image)
print("Colored Image written to file-system : ",status5)

save6 = (photo_path + '0g.png')
status6 = cv2.imwrite(save6, wane)
print("Colored Image written to file-system : ",status6)

#%%

# Measuring Properties

clusters = measure.regionprops(labeled_mask, img)
# measuring region properties of the image with Skimage

# These two lists are necessary to format the CSV file so that no column gets overwritten by a different column and the coordinates get written at the end under the names y,cord and x.cord
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
file = ("8_data.csv")
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


#%%
clusters = measure.regionprops(labeling, img)
# measuring region properties of the image with Skimage

# These two lists are necessary to format the CSV file so that no column gets overwritten by a different column and the coordinates get written at the end under the names y,cord and x.cord
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
file = ("8a_data.csv")
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




