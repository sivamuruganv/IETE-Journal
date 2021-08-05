import os
import numpy as np

import keras
from keras.applications.densenet import DenseNet201
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import load_model
from keras import backend as K

from io import BytesIO
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

import requests

K.set_learning_phase(0) #set the learning phase to not training

base_model =DenseNet201(weights='imagenet', include_top=False)

model = load_model('model.features.26-0.98.hdf5')

print(model.summary())

# Utility functions
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# Preprocess the input
# Rescale the values to the same range that was used during training 
def preprocess_input(x):
    x = img_to_array(x) / 255.
    return np.expand_dims(x, axis=0) 

# Prediction for an image path in the local directory
def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(224, 224)))

# Prediction for an image URL path
def predict_from_image_url(image_url):
    res = requests.get(image_url)
    im = Image.open(BytesIO(res.content))
    return predict_from_image_path(im.fp)
    
# Predict an image
def predict_image(im):
    x = preprocess_input(im)
    x = base_model.predict(x)
    pred = np.argmax(model.predict(x))
    return pred, classes[pred]

def grad_CAM(image_path):
    im = load_img(image_path, target_size=(224,224))
    x = preprocess_input(im)
    x = base_model.predict(x)
    pred = model.predict(x)
    
    # Predicted class index
    index = np.argmax(pred)
    
    # Get the entry of the predicted class
    class_output = model.output[:, index]
    
    # The last convolution layer in the model
    last_conv_layer = model.get_layer('conv2d_1')
    # Number of channels
    nmb_channels = last_conv_layer.output.shape[3]

    # Gradient of the predicted class with respect to the output feature map of the 
    # the convolution layer with nmb_channels channels
    grads = K.gradients(class_output, last_conv_layer.output)[0]   
    
    # Vector of shape (nmb_channels,), where each entry is the mean intensity of the gradient over 
    # a specific feature-map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Setup a function to extract the desired values
    iterate = K.function(model.inputs, [pooled_grads, last_conv_layer.output[0]])
    # Run the function to get the desired calues
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # Multiply each channel in the feature-map array by how important this channel is with regard to the 
    # predicted class
 
    for i in range(nmb_channels):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    # The channel-wise mean of the resulting feature map is the heatmap of the class activation.
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    # Normalize the heatmap betwen 0 and 1 for visualization
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
       
    # Read the image again, now using cv2
    img = cv2.imread(image_path)
    # Size the heatmap to the size of the loaded image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert to RGB
    heatmap = np.uint8(255 * heatmap)
    # Pseudocolor/false color a grayscale image using OpenCVs predefined colormaps
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
 
    # Superimpose the image with the required intensity
    superimposed_img = heatmap * 0.5 + img   
    
    # Write the image
    plt.figure(figsize=(24,12))
    cv2.imwrite('./tmp.jpeg', superimposed_img)
    plt.imshow(mpimg.imread('./tmp.jpeg'))
    plt.title(image_path)
    plt.show()

for i, c in enumerate(classes):
    folder = '/home/sivamurugan/deep-learning-models-master/224_OCT2017/val/' + c + '/'
    count = 1
    for file in os.listdir(folder):
        if file.endswith('.jpeg') == True:
            image_path = folder + file
            p, class_name = predict_from_image_path(image_path)
            if p == i:
                print(file, p, class_name)
            else:
                print(file, p, class_name, '**INCORRECT PREDICTION**')
                grad_CAM(image_path)
        count = count +1
        if count == 100:
            continue
