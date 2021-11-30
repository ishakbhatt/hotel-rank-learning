import os
import csv
import requests
import shutil
import pandas as pd
from src.utils import get_img_url_data_directory, get_temp_dir, get_structured_data_dir

def main():

    temp = get_temp_dir()
    df = pd.read_csv(os.path.join(get_structured_data_dir(), 'Hotel_metadata.csv'))

    # get file names into a list
    img_url_files = []
    img_url_data_path = get_img_url_data_directory()
    num_files = len(os.listdir(img_url_data_path))
    for i in range(num_files):
        filename = "image_part" + str(i+1) + ".csv"
        img_url_files.append(filename)

    # for each image
    for i in range(num_files):

        img_url_data_path = os.path.join(img_url_data_path, img_url_files[i])
        print("Image Part " + str(i + 1))

        with open(img_url_data_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            count = 0

            # skip header
            next(csvreader)

            # create package of image and label for each row
            for row in csvreader:
                count += 1
                if row[3] != '1' and row[3] != '11':
                    # only upload exterior images
                    continue

                # make a directory with hotelid as the name
                hotel_dir_name = 'hotel_' + row[0]
                os.makedirs(os.path.join(temp, hotel_dir_name), exist_ok=True)
                # look up rating with hotelid in hotel_metadata.csv
                if(len(df.loc[df['hotelid'] == int(row[0])]['star'].values) != 0):
                    star_rating = int(df.loc[df['hotelid'] == int(row[0])]['star'].values[0])

                    # write rating to CSV with rating+hotelid.csv as the name
                    hotel_csv = 'rating_' + row[0] + '.csv'
                    with open(hotel_csv, 'w') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(['star', str(star_rating)])

                    # move csv to package
                    shutil.move(os.path.join(os.getcwd(), hotel_csv), os.path.join(get_temp_dir(), hotel_dir_name, hotel_csv))

                    # extract image into directory
                    img_url = row[1]
                    img_name = os.path.basename(img_url)
                    response = requests.get(img_url, stream=True)
                    file = open(os.path.join(get_temp_dir(), hotel_dir_name, img_name), "wb")
                    file.write(response.content)
                    file.close()

                    # upload directory to S3
                    os.chdir(os.path.join(get_temp_dir(), hotel_dir_name))
                    os.system('aws s3 sync . s3://labeled-exterior-images/' + hotel_dir_name)
                    os.chdir('../../src/preprocessing/')
                    print("Uploading image and rating for " + img_name + " to S3 from row " + str(count) +"...")
                    # remove hotel directory
                shutil.rmtree(os.path.join(get_temp_dir(), hotel_dir_name))

if __name__ == "__main__":
    main()
