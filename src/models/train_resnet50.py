import csv
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50
from PIL import ImageFile
from src.utils import get_train_exterior_path, get_models_path, get_train_path, get_data_path, star_onehot_encode, is_corrupted
#from src.preprocessing.augment_image import augment_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(img_height, img_width, train_path, skip_deserialize=False):
    """
    :param img_height:
    :param img_width:
    :param train_path:
    :return:
    """
    labels = os.listdir(train_path)
    #label are 1star, ..., 5star. Image files are group into 5 folders, with folder name = star number 
    labels = [p for p in labels if p.endswith('star')]

    hotelid_image_mapping = pd.DataFrame(columns=['hotelid', 'image_serialized', 'star'])

    idx=0
    for label in labels:
        label_path = os.path.join(train_path, label)
        image_filenames = os.listdir(label_path)
        temp_star = label[0] # first char of '5star' is 5

        for image_filename in image_filenames:
            if(is_corrupted(image_filename, temp_star) == True):
                print("Skipping corrupted file ", image_filename, " from ", temp_star, " stars...")
                continue            
            temp_img = image.load_img(os.path.join(label_path, image_filename), target_size=(img_height, img_width))
            # image serialization
            temp_img = image.img_to_array(temp_img).astype('uint8').tobytes()
            temp_hotelid = int(image_filename[0 : image_filename.find('_')])
            new_row = pd.DataFrame([[temp_hotelid, temp_img, temp_star]], columns=hotelid_image_mapping.columns, index=[idx])
            hotelid_image_mapping = hotelid_image_mapping.append(new_row)
            idx += 1
    
    # shuffle image orders
    hotelid_image_mapping = hotelid_image_mapping.sample(frac=1)
    
    if (skip_deserialize==True):
        return None, None, hotelid_image_mapping
    else:
        X_train, y_train = deserialize_image(hotelid_image_mapping, img_height, img_width)
        return X_train, y_train, hotelid_image_mapping

def deserialize_image(hotelid_image_mapping, img_height, img_width):
    global parallel_deserialization
    X_train = list()
    train_label = []
    
    
    train_label = hotelid_image_mapping['star'].to_numpy(dtype='uint8', copy = True)
    y_train = star_onehot_encode(train_label)
    
    # image deserialization
    print("image deserialization...")
    num_images = hotelid_image_mapping['image_serialized'].count()
    def parallel_deserialization(idx):
        print("Deserializing for image: ", idx)
        temp_img = hotelid_image_mapping.at[idx, 'image_serialized']
        temp_deserialized_img = np.frombuffer(temp_img, dtype='uint8').reshape(img_height, img_width, 3)
        X_train.append(np.array(temp_deserialized_img))
        hotelid_image_mapping.drop(index=idx, inplace=True)
        return X_train

    # Distribute deserialization across cores
    from multiprocessing import Pool, cpu_count
    pool = Pool(cpu_count())
    X_train_list = pool.map(parallel_deserialization, range(num_images))
    X_train = pd.concat(X_train_list)

    X_train = np.asarray(X_train, dtype='float16')
    return X_train, y_train

def resnet50_model(num_classes):
    """
    :param num_classes:
    :return:
    """
    model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    x = Dropout(0.5)(model.output)
    #x = Dense(num_classes, kernel_regularizer='l2')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, x)
    
    # Train last few layers
    for layer in model.layers[:-19]:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])
    return model

# processing

if __name__ == '__main__':

    os.makedirs(os.path.join(get_data_path(), "models"), exist_ok=True)
    
    # training begin
    b_start = time.time()
    train_path = get_train_exterior_path()
    model_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')

    img_height = 210
    img_width = 280
    batch_size = 26
    epochs = 10

    X, Y, _ = load_images(img_height, img_width, train_path)
    num_classes = 5 # five star categories

    model = resnet50_model(num_classes)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   restore_best_weights=True,
                                   patience=7)

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[checkpointer],
                        verbose=1)

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
 
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
 
    # measure accuracy and F1 score 
    yhat_classes = model.predict(X_val)
    yhat_classes = np.argmax(yhat_classes,axis=1)
    y_true=np.argmax(Y_val,axis=1)
           
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, yhat_classes, average='weighted')
    print('Weighted F1 score: %f' % f1)
    # confusion matrix
    matrix = confusion_matrix(y_true, yhat_classes)
    print(matrix)

    
    print("Total used time : {} s.".format(time.time()-b_start))
