"""
Notes down names of corrupted images in a CSV that should be skipped in serialization.
"""
import csv
import os
from shutil import move
from os import listdir
from PIL import Image
from src.utils import get_train_exterior_path, get_corrupted_path

num_classes = 5
for i in range(num_classes):
    star = i + 1
    star_folder = str(star) + "star"
    corrupted_filenames = []
    for filename in listdir(os.path.join(get_train_exterior_path(), star_folder)):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(get_train_exterior_path(), star_folder, filename)) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError):
                corrupted_filenames.append(filename)
                print('Bad file:', filename) # print out the names of corrupt files
    csv_filename = star_folder + ".csv"
    with open(csv_filename, 'w') as file:
        write = csv.writer(file)
        write.writerow(corrupted_filenames)

    move(csv_filename, os.path.join(get_corrupted_path(), csv_filename))