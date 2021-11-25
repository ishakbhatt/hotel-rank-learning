import os
import csv
import shutil
import requests
#from src.navigation import get_train_path

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

if __name__ == "__main__":
    img_url_exterior = os.path.join(get_train_path(), "imageURL-exterior")
    star_number = input("Num stars to download for: ")
    print("Star: ", star_number)
    star_folder = star_number + "star"
    csv_name = star_number + "star.csv"
    count = 0
    with open(os.path.join(img_url_exterior, csv_name)) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        for row in csvreader:
            print("star: ", star_number, "row: ", str(count+1))
            count += 1
            img_url = row[1]
            img_name = row[0] + "_" + os.path.basename(img_url)
            response = requests.get(img_url, stream=True)
            file = open(img_name, "wb")
            file.write(response.content)
            file.close()
            #shutil.move(img_name, os.path.join(img_url_exterior, star_folder, img_name))
            os.system("aws s3 cp " + img_name + " s3://exterior-images-xstar/" + star_folder + "/" + img_name)
            os.remove(img_name)

