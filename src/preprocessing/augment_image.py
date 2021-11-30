import sys
import os
import numpy as np
from PIL import Image, ImageEnhance
import shutil
sys.path.append("..")
from utils import get_train_path, get_train_exterior_path, is_corrupted
sys.path.remove("..")

# augmentation

def augment_examples(examples_path, temp_dir, aug_type="brightness", augment_cap=-1, enhancement=1.1):
    examples = os.listdir(examples_path)
    x = len(examples)
    count = 0
    for example in examples:
        if (is_corrupted(example, os.path.basename(examples_path)[0]) == True):
            break
        if (count == augment_cap and augment_cap != -1):
            break
        _, ext = os.path.splitext(example)
        if(ext):
            img = Image.open(os.path.join(examples_path, example))
            new_img = img
            filter = ImageEnhance.Brightness(img)
            filename_addition = "_brightened" + str(enhancement)
            if aug_type == "sharpness":
                print("sharpening...")
                filter = ImageEnhance.Sharpness(img)
                filename_addition = "_sharpened" + str(enhancement)
            elif aug_type == "saturate":
                print("saturating...")
                filter = ImageEnhance.Color(img)
                filename_addition = "_saturated" + str(enhancement)
            elif aug_type == "horizontal_flip":
                print("flipping horizontally...")
                data = np.asarray(img)
                data = np.fliplr(data)
                new_img = Image.fromarray(data)
                filename_addition = "_flipped" 
            elif aug_type == "contrast":
                print("contrasting...")
                filter = ImageEnhance.Contrast(img)
                filename_addition = "_contrasted" + str(enhancement)
            else:
                print("brightening...")

            if aug_type != "horizontal_flip":
                new_img = filter.enhance(enhancement)
            
            new_img_name = os.path.splitext(example)[0] + filename_addition + ".jpg"
            new_img.save(os.path.join(temp_dir, new_img_name))
            count += 1
    if temp_dir == os.path.join(get_train_path(), "temp_5star"):
        breakpoint()
    c = count

if __name__ == "__main__":
    # augment 1star from 1k to 11k
    print("Augmenting 1 star...")
    temp_1star_path = os.path.join(get_train_path(), "temp_1star")
    os.makedirs(temp_1star_path, exist_ok=True)
    examples_path = os.path.join(get_train_exterior_path(), "1star")

    augment_examples(examples_path, temp_1star_path, aug_type="saturate", enhancement=3)
    augment_examples(examples_path, temp_1star_path, enhancement=0.6)
    augment_examples(examples_path, temp_1star_path, aug_type="saturate", enhancement=1.7)

    # augment 5star from 16K to 26K
    print("Augmenting 5 star...")
    temp_5star_path = os.path.join(get_train_path(), "temp_5star")
    os.makedirs(temp_5star_path, exist_ok=True)
    examples_path = os.path.join(get_train_exterior_path(), "5star")
    #augment_examples(examples_path, temp_5star_path, aug_type="horizontal_flip")
    augment_examples(examples_path, temp_5star_path, aug_type="sharpness", enhancement=3)
    #augment_examples(examples_path, temp_1star_path, aug_type="horizontal_flip")
    augment_examples(examples_path, temp_5star_path, aug_type="contrast", enhancement=0.4)
    augment_examples(examples_path, temp_5star_path, enhancement=1.4)
    augment_examples(examples_path, temp_5star_path, aug_type="saturate", enhancement=3)
    augment_examples(examples_path, temp_5star_path, aug_type="contrast", enhancement=1.3)
    augment_examples(examples_path, temp_5star_path, aug_type="sharpness", enhancement=1.4)
    augment_examples(examples_path, temp_5star_path, enhancement=0.5)
    augment_examples(examples_path, temp_5star_path, aug_type="saturate", enhancement=1.6)