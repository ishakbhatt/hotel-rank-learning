import sys
import os
import random
import numpy as np
from PIL import Image, ImageEnhance
sys.path.append("..")
from utils import get_train_exterior_path, is_corrupted
sys.path.remove("..")

# augmentation
def augment_examples(examples_path, aug_type="brightness", aug_cap=-1, enhancement=1.1):
    examples = os.listdir(examples_path)
    random.shuffle(examples)
    count = 0
    for example in examples:
        if (is_corrupted(example, os.path.basename(examples_path)[0]) == True):
            break
        if (count == aug_cap and aug_cap != -1):
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
            if new_img.mode in ("RGBA", "P"): new_img = new_img.convert("RGB")
            new_img.save(os.path.join(examples_path, new_img_name))
            count += 1
    print("Added " + str(count) + " images to class " + os.path.basename(examples_path))

if __name__ == "__main__":

    # set amount to augment by for each class
    num_classes = 5
    aug_caps = [10000, 2000, 2000, 2000, 10000]
    for i in range(num_classes):
        aug_cap = -1
        if aug_caps[i] != -1:
            aug_cap = aug_caps[i] / 10

        examples_path = os.path.join(get_train_exterior_path(), str(i+1) + "star")
        print("Augmenting ", str(i+1), " star...")
        augment_examples(examples_path, aug_type="saturate", enhancement=1.5, aug_cap=aug_cap)
        augment_examples(examples_path, enhancement=0.6, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="sharpness", enhancement=1.4, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="contrast", enhancement=0.4, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="horizontal_flip", aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="saturate", enhancement=2, aug_cap=aug_cap)
        augment_examples(examples_path, enhancement=0.4, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="sharpness", enhancement=1.2, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="sharpness", enhancement=1.3, aug_cap=aug_cap)
        augment_examples(examples_path, aug_type="contrast", enhancement=0.5, aug_cap=aug_cap)