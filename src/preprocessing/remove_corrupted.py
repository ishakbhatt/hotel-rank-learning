import os

import sys
sys.path.append("..")
from utils import get_train_exterior_path, is_corrupted
sys.path.remove("..")

folders = ["5star"]
stars = ["5"]
idx = 0
for folder in folders:
    print(folder)
    examples = os.listdir(os.path.join(get_train_exterior_path(), folder))
    for example in examples:
        if is_corrupted(example, stars[idx]) == True:
            os.remove(os.path.join(get_train_exterior_path(), folder, example))
            print("remove corrupted example ", example, "...")
        else:
            print(example)
    idx +=1