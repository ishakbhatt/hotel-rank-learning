import os, sys, time, shutil, numpy as np
import matplotlib.pyplot as plt
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
sys.path.append("..")
from utils import get_train_exterior_path, get_models_path, load_image_uri, get_data_path
sys.path.remove("..")

ImageFile.LOAD_TRUNCATED_IMAGES = True
results = []

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
    epochs = 50
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
        class_mode="categorical",
        shuffle=False,
    )
    
    model = resnet50_model(num_classes)
    
    # training begin
    b_start = time.time()
    ckpt_path = os.path.join(get_models_path(), 'resnet50_ResNet50_v1.h5')
    model.load_weights(ckpt_path)
    checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=5)
    
    history = model.fit(train_generator, validation_data = train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = valid_generator.n//valid_generator.batch_size,
                    epochs=epochs, callbacks=[checkpointer, early_stopping], verbose=1)
    score = model.evaluate_generator(test_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


 # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
 
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.tight_layout()
    plt.savefig("CNN_Accuracy_Loss.png")
    shutil.move("CNN_Accuracy_Loss.png", os.path.join(get_data_path(), "analysis", "CNN", "CNN_Accuracy_Loss.png"))
 
    # measure accuracy and F1 score 
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
               
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(test_generator.classes, y_pred)
    print('Accuracy: %f' % accuracy)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test_generator.classes, y_pred, average='weighted')
    print('Weighted F1 score: %f' % f1)
    # confusion matrix
    matrix = confusion_matrix(test_generator.classes, y_pred)
    print(matrix)  
    print("Total used time : {} s.".format(time.time()-b_start))
