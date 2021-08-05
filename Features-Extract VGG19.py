
from __future__ import print_function
from keras.applications.vgg19  import VGG19
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import h5py

conv_base = VGG19(weights='imagenet', include_top=False)

train_dir ='/home/sivamurugan/deep-learning-models-master/224_OCT2017/train/'
validation_dir = '/home/sivamurugan/deep-learning-models-master/224_OCT2017/test/'

def extract_features(file_name, directory, key, sample_count, target_size, batch_size, class_mode='categorical'):
    h5_file = h5py.File(file_name, 'w')
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(directory, target_size=target_size,
        batch_size=batch_size, class_mode=class_mode)
    
    samples_processed = 0
    batch_number = 0
    if sample_count == 'all':
        sample_count = generator.n
          
    print_size = True
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        
        if print_size == True:
            print_size = False
            print('Features shape', features_batch.shape)
            
        samples_processed += inputs_batch.shape[0]
        h5_file.create_dataset('features-'+ str(batch_number), data=features_batch)
        h5_file.create_dataset('labels-'+str(batch_number), data=labels_batch)
        batch_number = batch_number + 1
        print("Batch:%d Sample:%d\r" % (batch_number,samples_processed), end="")
        if samples_processed >= sample_count:
            break
  
    h5_file.create_dataset('batches', data=batch_number)
    h5_file.close()
    return

extract_features('/home/sivamurugan/deep-learning-models-master/VGG19/VGG19/train.h5', train_dir, key='train', 
                                    sample_count='all', batch_size=100, target_size=(224,224))

extract_features('/home/sivamurugan/deep-learning-models-master/VGG19/VGG19/val.h5', validation_dir, key='val', 
                                    sample_count='all', batch_size=100, target_size=(224,224))
