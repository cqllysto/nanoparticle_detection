# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:10:56 2022

@author: aidan
"""
import os
import image_slicer

#%%
tiles = image_slicer.slice("C:/Users/aidan/Documents/Nanoparticle_Detection/Images/5.jpg", 2, save=False)
image_slicer.save_tiles(tiles, directory="C:/Users/aidan/Documents/Nanoparticle_Detection/Data",\
                        prefix='ptvg')


#%%


main_folder = "C:/Users/aidan/Documents/Nanoparticle_Detection"
photo_folder = "C:/Users/aidan/Downloads/Particles/Particles"
data_folder = "C:/Users/aidan/Documents/Nanoparticle_Detection/Data"



for count, filename in enumerate(os.listdir(photo_folder)):
    dst = f"{str(count)}.jpg"
    src =f"{photo_folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{photo_folder}/{dst}"
    os.rename(src, dst)
#    tiles = image_slicer.slice(f"C:/Users/aidan/Documents/Nanoparticle_Detection/Images/{str(count)}.jpg", 5, save=False)
#    image_slicer.save_tiles(tiles, directory="C:/Users/aidan/Documents/Nanoparticle_Detection/Data",\
#                            prefix=f'{str(count)}a.jpg')