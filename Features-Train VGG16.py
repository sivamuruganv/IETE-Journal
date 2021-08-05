from __future__ import print_function


import keras
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import conv2d_bn


from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt


def features_from_file(path, ctx):
    h5f = h5py.File(path, 'r')
    batch_count = h5f['batches'].value
    print(ctx, 'batches:', batch_count)       
    
    def generator():
        while True:
            for batch_id in range(0, batch_count):
                X = h5f['features-' + str(batch_id)]
                y = h5f['labels-' + str(batch_id)]
                yield X, y
            
    return batch_count, generator()

train_steps_per_epoch, train_generator = features_from_file('/home/sivamurugan/deep-learning-models-master/VGG16/VGG16/train-ALL.h5', 'train')
validation_steps, validation_data = features_from_file('/home/sivamurugan/deep-learning-models-master/VGG16/VGG16/val-ALL.h5', 'val')

np.random.seed(7)
inputs = Input(shape=(7, 7, 512))
x = conv2d_bn(inputs, 64, 1, 1)
x = Dropout(0.5)(x)
x = Flatten()(x)
outputs  = Dense(4, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizers.adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())

# Setup a callback to save the best model
callbacks = [ 
    ModelCheckpoint('/home/sivamurugan/deep-learning-models-master/VGG16/VGG16/model.features.{epoch:02d}-{val_acc:.2f}.hdf5', 
                 monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1),
             
    ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.5, patience=5, min_lr=0.00005)
            ]

history = model.fit_generator(
            generator=train_generator, steps_per_epoch=train_steps_per_epoch,  
            validation_data=validation_data, validation_steps=validation_steps,
            epochs=100, callbacks=callbacks)

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,8))
    plt.plot(epochs, acc, 'b', color='red' , label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.figure(figsize=(12,8))
    plt.plot(epochs, val_acc, 'b', color='red', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()


    plt.figure(figsize=(12,8))
    plt.plot(epochs, loss, 'b',color='red', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    
    plt.figure(figsize=(12,8))
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
    plt.title('Validation loss')
    plt.legend()


    plt.show()
    return acc, val_acc, loss, val_loss


acc, val_acc, loss, val_loss = plot_history(history)

