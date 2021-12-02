# Seeing Stars: A Computer Vision-Centered Approach for Star Rating in the Hospitality Industry
With over 10 million hotels globally, there is a need for travel booking websites to provide accurate
and reliable information to visitors. Star rating is most frequently used as a filtering criterion, but is
unreliable given the absence of commonly accepted standards for star rating assignment. Manual
human verification can be subjective and incurs high operating costs. Several major third-party
distribution platforms, e.g., Booking.com, therefore let hotel owners self-report their own star ratings,
with highly inconsistent results.

Our objective is to create a computer-vision-assisted machine learning model that can more accurately
assign hotel star ratings using images and meta-data (e.g. pricing, facilities). This promises a cheaper
and more objective methodology in assigning hotel star ratings.

The full project proposal can be found [here](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/project_proposal/CS_230_Project_Proposal__Ye__Zhuo__Bhatt_.pdf).

## Data Preparation

### Image Scraping
Images can be scraped from a CSV with images, hotelid, and star category using [this script](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/src/preprocessing/image_scraper.py). We are unable to provide the CSV with image links at this time.  

### Handling Corrupted Images
To check for corrupted images and note down in a CSV, run `check_for_corrupted.py` from `src/preprocessing`.  

To remove corrupted images, run `remove_corrupted.py` from `src/preprocessing`.

### Data Augmentation  
We implemented our own (naive) data augmentation method which can be found [here](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/src/preprocessing/augment_image.py). To augment any of the five classes by a certain amount, the respective value in the list on [line 60](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/src/preprocessing/augment_image.py#L60) can be changed. For example, to increase the number of images for three classes (2, 3, 4 stars) by 2000 and for two classes (1 and 5 stars) by 10,000, `aug_caps` would be set to `[10000, 2000, 2000, 2000, 10000]`.  

Note that this is a naive implementation since the amount of memory used would increase by saving more images in the dataset directories. For a future iteration, we would like to reduce memory usage by moving the data augmentation step online using the ImageDataGenerator API rather than doing data augmentation offline using our own data augmentation method.

To augment the data, run `augment_image.py` from `src/preprocessing`. The images will get saved directly in the class directories in `data/train/exterior/`. You can run `ls | wc -l` in each of the class directories to verify that the classes increased by the values specified in the `aug_caps` list.  

## Training
We developed two supervised learning models that apply the ResNet-50 architecture on a labeled dataset consisting of hotel images from each of the five star categories. Both models classify hotels into their respective ratings using **exterior image data**. We also developed a DNN that uses two layers with LeakyReLU activation functions and softmax regression as the last layer. We then developed a combined model that uses weights from our DNN and CNN to learn from both structured and unstructured data to correctly predict the hotel star rating.   

### About ResNet50
ResNet stands for Residual Network and ResNet-50 is a CNN that is 50 layers deep. This architecture was first introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in a 2015 research paper, which you can read [here](https://arxiv.org/abs/1512.03385).

### ResNet50-based CNN: Loading images into dataframe  

Our [first method](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/src/models/train_resnet50.py) loads images as PIL images directly with the HotelID and star into a large dataframe. With a large dataset (ex. 24K hotels with >100K images), loading images into a dataframe becomes expensive in terms of memory.  

The model can be trained directly from `train_resnet50.py` from the `src/models` directory (do not train from root). 

### ResNet50-based CNN: Loading images using ImageDataGenerator API

We then developed a second iteration of our CNN. Our [second method](https://github.com/ishakbhatt/hotel-rank-learning/blob/main/src/models/train_images.py) loads the image paths (instead of the actual PIL image objects) with the HotelID and star rating into a large dataframe. We leverage the [ImageDataGenerator API](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator), which generates batches of tensor image data for training, validation, and test sets. For large datasets, this API works very well because (for each dataset - train, validation, test) the ImageDataGenerator feeds the data gradually to the neural network without keeping it in memory. 

The model (`train_images.py`) can be run from the `src/models` directory (do not run from root).

### Hotel Metadata DNN  
This model classifies hotels into their respective ratings using **hotel metadata**.    

The model (`train_structured.py`) can be run from the `src/models` directory (do not run from root).

### Combined Model

The model (`train_full_model.py`) uses weights from `train_structured.py` and `train_resnet50.py` and can be run from the `src/models` directory (do not run from root).  

## Analysis  

### Class Distribution  
Run `class_dist.py` from `src/preprocessing`. The class distribution (without augmentation) can be found in `data/data_analysis`.

### Evaluation Metrics  
The metrics for `train_images.py` and `train_full_model.py` can be found in `data/analysis/CNN` and `data/analysis/Combined`, respectively. We ran `train_images.py` three different times on images for 26,000 hotels:   
* no augmentation
* augmentation on just 1 star and 5 star classes
* augmentation on all classes: To avoid biasing augmented images towards 1 star and 5 star classes, we augmented all the classes, augmenting the smaller classes (1 star and 5 star) more than the others.  

## Libraries
`pillow`    
`numpy`    
`tensorflow`    
`shutil`
`multiprocessing`  
`time`  
`sys`      
`random`  
`requests`  
`dask`  
`zipfile`  
`os`  
`matplotlib`  
`sklearn`
`pandas`
