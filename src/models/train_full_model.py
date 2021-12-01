import os, sys, shutil, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU, Dropout
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
sys.path.append("..")
from utils import get_train_exterior_path, get_data_path, get_models_path
sys.path.remove("..")
from train_resnet50 import load_images, deserialize_image
from train_structured import load_metadata

def align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width):
    """
    align the inputs of the CNN(images) and DNN(metadata) models, such that two inputs have identical sample size
    and the samples of the same index position across two input arrays carry the same hotelid.
    Referential relationship is that metadata:images = 1:many
    """
    #inner join image and metadata on hotelid
    metaX = pd.concat([metaX, meta_hotelids.astype('int32')], axis=1)
    df_merged = hotelid_image_mapping.merge(metaX, how='inner', left_on='hotelid', right_on='hotelid')
        
    #split train and test sets
    df_merged_train, df_merged_val = train_test_split(df_merged, test_size=0.2, random_state=0)
    df_merged_train.reset_index(inplace=True)
    df_merged_val.reset_index(inplace=True)
    
    #after the alignment join, split the merged dataset back into image and metadata
    metaX_train = df_merged_train[metaX.columns].copy(deep=True).astype('float16')
    metaX_val = df_merged_val[metaX.columns].copy(deep=True).astype('float16')
    metaX_train = np.nan_to_num(np.asarray(metaX_train.drop(['hotelid'], axis = 1)))
    metaX_val = np.nan_to_num(np.asarray(metaX_val.drop(['hotelid'], axis = 1)))
    
    #after the alignment join, split the merged dataset back into image and metadata
    imageX_train, Y_train = deserialize_image(df_merged_train, img_height, img_width)
    imageX_val, Y_val = deserialize_image(df_merged_val, img_height, img_width)

    return imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val
    
if __name__ == '__main__':
    img_height = 225
    img_width = 300
    channels = 3
    batch_size = 1
    epochs = 50
    num_classes = 5
    
    train_path = get_train_exterior_path()
    
    _, _, hotelid_image_mapping = load_images(img_height, img_width, train_path, skip_deserialize=True)
    metaX, _, meta_hotelids = load_metadata("hotel_meta_processed.csv")
    imageX_train, metaX_train, imageX_val, metaX_val, Y_train, Y_val = align_model_inputs(hotelid_image_mapping, metaX, meta_hotelids, img_height, img_width)
    
    input_CNN = Input(shape=(img_height, img_width, channels))
    input_DNN = Input(shape=(metaX_train.shape[1]))
    
    CNN_base = ResNet50(weights='imagenet', pooling='avg', include_top=False)
    CNN_dropout = Dropout(0.4, name = 'image_dropout')(CNN_base.output)
    CNN_dense1 = Dense(512, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name='image_dense1')(CNN_dropout)
    CNN_dense2 = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name='image_dense2')(CNN_dense1)
    
    #dnn_base = Sequential()
    dnn_layer1 = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name = 'meta_dense1', input_shape=(metaX.shape[1],))(input_DNN)
    dnn_layer2 = (Dense(64, activation=LeakyReLU(alpha=0.1), kernel_initializer='he_normal', name = 'meta_dense2'))(dnn_layer1)
    
    #cnn_model = Model(inputs=CNN_base.inputs, outputs=CNN_dense4)
    #dnn_model = Model(inputs=dnn_base.inputs, outputs=dnn_base.get_layer('pre_output_layer'))
    
    # Train last few layers
    for layer in CNN_base.layers[:-19]:
        layer.trainable = False
    
    # Concatenate
    concat = Concatenate(name='cancat_layer')([CNN_dense2, dnn_layer2])

    # output layer input_shape=(None, concat.shape[-1])
    output = Dense(units=num_classes, activation='softmax')(concat)
    
    full_model = Model(inputs=[input_DNN, CNN_base.inputs], outputs=[output])
    full_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    ckpt_path = os.path.join(get_models_path(), 'full_model.h5')
    full_model.load_weights(ckpt_path)
    checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=7)
    history = full_model.fit([metaX_train, imageX_train], Y_train,
                  validation_data=([metaX_val, imageX_val], Y_val),
                  callbacks = [checkpointer, early_stopping],
                  epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1)

    
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
 
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("FullModel_Loss_Accuracy.png")
    shutil.move("FullModel_Loss_Accuracy.png", os.path.join(get_data_path(), "analysis", "Combined", "FullModel_Loss_Accuracy.png"))

    # measure accuracy and F1 score 
    yhat_classes = full_model.predict([metaX_val, imageX_val])
    yhat_classes = np.argmax(yhat_classes, axis=1)
    y_true = np.argmax(Y_val, axis=1)
           
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, yhat_classes, average='weighted')
    print('Weighted F1 score: %f' % f1)
    # confusion matrix
    matrix = confusion_matrix(y_true, yhat_classes)
    print(matrix)
    