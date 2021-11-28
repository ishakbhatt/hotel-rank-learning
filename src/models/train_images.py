import csv
import os
import time
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet import ResNet50
from PIL import ImageFile
from src.utils import load_image_uri
#from src.utils import get_train_exterior_path, get_models_path, get_train_path, get_data_path, star_onehot_encode, is_corrupted
#from src.preprocessing.augment_image import augment_data

ImageFile.LOAD_TRUNCATED_IMAGES = True
results = []

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

def get_corrupted_path():
    """
    Return the path to directory containing csvs of corrupted data.
    :return: corrupted
    """
    os.chdir("../../data/corrupted")
    corrupted = os.path.join(os.getcwd())
    os.chdir("../../src/models")

    return corrupted

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

def resnet50_model(num_classes):
    """
    :param num_classes:
    :return:
    """
    model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    x = Dropout(0.3, name='image_dropout')(model.output)
    #x = Dense(num_classes, kernel_regularizer='l2')(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name='image_dense1')(x)
    x = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name='image_dense2')(x)
    x = Dense(num_classes, activation='softmax', name='image_softmax')(x)
    model = Model(model.input, x)
    
    # Train last few layers
    for layer in model.layers[:-19]:
        layer.trainable = False

    optmz = optimizers.Adam(learning_rate=0.001, epsilon=1e-8)
    model.compile(optimizer=optmz, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])
    return model

# processing

if __name__ == '__main__':
    img_height = 225
    img_width = 300
    batch_size = 32
    epochs = 1
    num_classes = 5 # five star categories
    
    os.makedirs(os.path.join(get_data_path(), "models"), exist_ok=True)
    train_path = get_train_exterior_path()
    
    train_image_uri, valid_image_uri, test_image_uri = load_image_uri(train_path)
    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_image_uri,
        x_col="image_uri", y_col="star",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_image_uri,
        x_col="image_uri", y_col="star",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_image_uri,
        x_col="image_uri", y_col="star",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
    )
    
    model = resnet50_model(num_classes)
    
    # training begin
    b_start = time.time()
    ckpt_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')
    model.load_weights(ckpt_path)
    checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=5)
    
    history = model.fit_generator(train_generator, validation_data = train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = valid_generator.n//valid_generator.batch_size,
                    epochs=epochs, callbacks=[checkpointer, early_stopping], verbose=1)
    breakpoint()
    score = model.evaluate_generator(test_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


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
    yhat_classes = model.predict_generator(test_generator, steps = len(test_generator.filenames))
    yhat_classes = np.argmax(yhat_classes)
    y_true = np.argmax(test_generator.get_classes(test_image_uri, 'star'), axis=1)
           
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
