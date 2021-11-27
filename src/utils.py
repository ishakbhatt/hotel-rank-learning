import os
import csv
import shutil
import numpy as np

"""
Model output utilities.
"""

def star_onehot_encode(stars):
    """
    :param stars: 1D array
    :return: one-hot encoded star ratings
    """
    # one hot encode
    num_class = 5 #from 1 star to 5 stars
    onehot_encoded = list()
    for star in stars:
        encoded = np.zeros(num_class)
        encoded[star-1] = 1
        onehot_encoded.append(encoded)

    return np.array(onehot_encoded)

"""
AWS Utilities.
"""

def refactor_into_label_directories():
    """
    Refactors img data from hotel directories into label directories when downloading from labeled-exterior-images.
    :return: void
    """
    os.makedirs(os.path.join(get_train_path(), "exterior2"), exist_ok=True)
    count = 1
    hotel_dirs = os.listdir(get_train_exterior_path())
    for hotel_dir in hotel_dirs:
        hotel_files = os.listdir(os.path.join(get_train_exterior_path(), hotel_dir))
        for file in hotel_files:
            if(os.path.splitext(file)[1][1:] == "csv"):
                continue
            with open(os.path.join(get_train_exterior_path(), hotel_dir,
                                   hotel_files[len(hotel_files)-1])) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=",")
                for star in csvreader:
                    shutil.copy(os.path.join(get_train_exterior_path(), hotel_dir, file),
                                os.path.join(get_train_path(), "exterior2", str(star[1]) + "star"))
                    print("count: " + str(count))
                    count += 1
        os.remove(os.path.join(get_train_exterior_path(), hotel_dir))

def download_from_aws(num_images):
    s3 = boto3.resource('s3')
    corrupted = os.listdir(os.path.join(get_data_path(), "Corrupted"))
    count = 0
    labeled_exterior_images = s3.Bucket('labeled-exterior-images')
    for object_sum in labeled_exterior_images.objects.filter(Prefix=""):
        if(count == num_images):
            break
        if(corrupted.count(os.path.basename(object_sum.key)) != 0):
            continue
        os.system("aws s3 sync s3://labeled-exterior-images/" + os.path.dirname(object_sum.key) + " " + os.path.join(get_train_exterior_path(), os.path.dirname(object_sum.key)))
        print("num images downloaded: " + str(count))
        count +=1

"""
Navigate through directories and files.
"""

"""PREPROCESSING"""
def get_corrupted_path():
    """
    Return the path to directory containing csvs of corrupted data.
    :return: corrupted
    """
    os.chdir("../../data/train/corrupted")
    corrupted = os.path.join(os.getcwd())
    os.chdir("../../../src/models")

    return corrupted

def get_img_url_data_directory():
    """
    Return the URL data directory.
    :return: img_url_data_path
    """
    # change directory
    os.chdir("../../url_data/image_urls")

    # get directory
    img_url_data_path = os.path.join(os.getcwd())
    print(img_url_data_path)

    os.chdir("../../src/preprocessing")

    return img_url_data_path

def get_temp_dir():
    """
    Return the temp directory which stores temporary labeled packages.
    :return: temp_data_path
    """
    # change directory
    os.chdir("../../temp")

    # get directory
    temp_data_path = os.path.join(os.getcwd())
    print(temp_data_path)

    os.chdir("../src/preprocessing")

    return temp_data_path

"""TRAINING & INFERENCE"""
def get_train_path():
    """
    Return the path to training data directories.
    :return: train_path
    """
    os.chdir("../../data/train/")
    train_path = os.path.join(os.getcwd())
    print(train_path)
    os.chdir("../../src/models")

    return train_path

def get_models_path():
    """
    Return the models path which stores the model checkpoint at a frequency.
    :return: models_path
    """
    os.chdir("../../data/models/")
    models_path = os.path.join(os.getcwd())
    print(models_path)
    os.chdir("../../src/models")

    return models_path

def get_train_exterior_path():
    """
    Return the path to exterior training data.
    :return:
    """
    os.chdir("../../data/train/exterior")
    exterior_path = os.path.join(os.getcwd())
    print(exterior_path)
    os.chdir("../../../src/models")

    return exterior_path

def get_data_path():
    """
    Return the path to exterior training data.
    :return:
    """
    os.chdir("../../data/")
    data_path = os.path.join(os.getcwd())
    print(data_path)
    os.chdir("../src/models")

    return data_path

def get_structured_data_path():
    """
    Return the path to structured data.
    :return: structured_data_path
    """
    os.chdir("../../data/url_data/structured_hotel_data")
    structured_data_path = os.path.join(os.getcwd())
    os.chdir("../../../src/models/")
    return structured_data_path

"""
Utilities for training data.
"""

def is_corrupted(filename, star):
    corrupted_path = get_corrupted_path()
    corrupted_list = []
    file = open(os.path.join(corrupted_path, star+"star"+".csv"), "r")
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        corrupted_list.append(row)
    if filename in corrupted_list[0]:
        return True
    return False