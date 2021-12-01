import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
sys.path.append("..")
from utils import get_models_path
sys.path.remove("..")

def get_structured_data_path():
    """
    Return the path to structured data.
    :return: structured_data_path
    """
    os.chdir("../../data")
    structured_data_path = os.path.join(os.getcwd())
    os.chdir("../src/models")
    return structured_data_path

def load_metadata(filename):
    raw_structured_data = pd.read_csv(os.path.join(get_structured_data_path(), filename))
    #shuffle
    raw_structured_data = raw_structured_data.sample(frac=1)
    
    #star{1,2,3,4,5}-> class_indices {0,1,2,3,4}
    for i in range(raw_structured_data["star"].count()):
        star = str(int(raw_structured_data.at[i, "star"]) - 1)
        raw_structured_data.at[i, "star"] = star
    
    # input
    x1 = raw_structured_data[["numReviewers", "roomRating", "serviceRating",
                            "value4moneyRating", "locatioRating", "roomquantity",
                            "minArea", "isStarValidated", "order_cnt", "roomnight_cnt",
                            "avg_room_price", "avg_person_price", "maxprice", "minprice",
                            "adr1", "adr2", "adr3", "adr4", "adr5", "avg_person_price_bycity", "pic_hq_ratio",
                            "pic_apperance_ratio", "pic_public_ratio", "pic_meeting_ratio",
                            "pic_restaurant_ratio", "pic_leisure_ratio", "pic_service_ratio",
                            "is_adr2_adjusted", "is_adr3_adjusted",
                            "is_adr4_adjusted", "is_adr5_adjusted"]]
    x2 = raw_structured_data[["openyear", "renovationyear", "gym", "executive_lounge",
                            "indoor_swimming_pool", "bathrobe", "laundry_service", "X24h_frontdesk",
                            "conference_hall", "luggage_storage", "roomcleaneddaily", "outdoor_swimming_pool"]]
    hotelids = raw_structured_data[["hotelid"]]
    
    encoder = OneHotEncoder(handle_unknown='error')
    x2_onehot = pd.DataFrame(encoder.fit_transform(x2).toarray())

    x = pd.concat([x1, x2_onehot], axis=1)

    # output
    y = raw_structured_data["star"]

    return x, y, hotelids

def train_linear_model(x_train, y_train, x_test):
    # define model & train
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])
    # inference
    predictions = np.round(model.predict(x_test))
    predictions[predictions>5] =5
    predictions[predictions<1] =1

    # metrics
    metrics.mean_absolute_error(y_test, predictions)
    metrics.mean_squared_error(y_test, predictions)
    np.sqrt(metrics.mean_squared_error(y_test, predictions))
    correct = predictions == y_test
    acc = sum(correct)/len(predictions)
    print('Test Accuracy: %.3f' % acc)
    
    return model

def train_DNN_model(x_train, y_train, x_test, y_test, epochs, batch_size):
    model = DNN_model((x_train.shape[1],))
    # compile the model
    optmz = optimizers.Adam(learning_rate=0.0002, epsilon=1e-8)
    model.compile(optimizer=optmz, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    # fit the model
    ckpt_path = os.path.join(get_models_path(), 'structured.h5')
    model.build(x_train.shape)
    model.load_weights(ckpt_path)
    checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=12)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,  
                        callbacks=[checkpointer, early_stopping], verbose=1)
    
    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

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
    yhat_classes = model.predict(x_test)
    yhat_classes = np.argmax(yhat_classes,axis=1)
           
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Test Accuracy: %f' % accuracy)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes, average='weighted')
    print('Weighted F1 score: %f' % f1)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    
    return model
 
def DNN_model(input_shape):
     # define model
    model = Sequential()
    model.add(Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', input_shape=input_shape, name='meta_dense1'))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name='meta_dense2'))
    model.add(Dense(5, activation='softmax', name='meta_softmax'))
        
    return model

if __name__ == "__main__":
    x, y, _ = load_metadata("hotel_meta_processed.csv")
    
    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    #handling null(if any)
    x_train = np.nan_to_num(x_train)
    y_train = np.nan_to_num(y_train)
    x_test = np.nan_to_num(x_test)
    y_test = np.nan_to_num(y_test)

    #linear_model = train_linear_model(x_train, y_train, x_test)
    train_DNN_model(x_train, y_train, x_test, y_test, 100, 32)
    
