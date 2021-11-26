# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:31:36 2021

@author: Bo Xian Ye
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.navigation import get_train_exterior_path
from train_resnet50 import load_images, deserialize_image, resnet50_model
from train_structured import load_metadata, train_DNN_model

def align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width):
    """
    align the inputs of the CNN(images) and DNN(metadata) models, such that two inputs have identical sample size
    and the samples of the same index position across two input arrays carry the same hotelid.
    Referential relationship is that metadata:images = 1:many
    """
    #inner join image and metadata on hotelid
    metaX = pd.concat([metaX, meta_hotelids.astype('int32')], axis=1)
    df_merged = hotelid_image_mapping.merge(metaX, how='inner', left_index=True, right_on='hotelid')
    #print (df_merged.count())
        
    #split train and test sets
    df_merged_train, df_merged_val = train_test_split(df_merged, test_size=0.2, random_state=0)
    
    #after the alignment join, split the merged dataset back into image and metadata
    imageX_train, Y_train = deserialize_image(df_merged_train, img_height, img_width)
    imageX_val, Y_val = deserialize_image(df_merged_val, img_height, img_width)
    metaX_train = df_merged_train[metaX.columns]
    metaX_val = df_merged_val[metaX.columns]
    metaX_train.drop(['hotelid'], axis = 1)
    metaX_val.drop(['hotelid'], axis = 1)

    return imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val
    
if __name__ == '__main__':
    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 100
    num_classes = 5
    
    train_path = get_train_exterior_path()
    
    _, _, hotelid_image_mapping = load_images(img_height, img_width, train_path, skip_deserialize=True)
    metaX, _, meta_hotelids = load_metadata("hotel_meta_processed.csv")
    imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val = align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width)
    
    DNN_model = train_DNN_model(metaX_train, Y_train, metaX_val, Y_val, epochs, batch_size)
    CNN_model = resnet50_model(num_classes)
    CNN_model.fit(imageX_train, Y_train,
                  validation_data=(imageX_val, Y_val),
                  epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1)
    