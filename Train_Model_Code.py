import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import os
# the gestures
CATEGORY_MAP = {
    "scroll" : 0,
    "zoomin" : 1,
    "zoomout" : 2,
    "rotatecw" : 3,
    "rotateccw" : 4,
    "empty" : 5
}
# model parameters
def def_model_param():
    GESTURE_CATEGORIES = len(CATEGORY_MAP)
    base_model = Sequential()
    base_model.add(SqueezeNet(input_shape=(225, 225, 3), include_top=False))
    base_model.add(Dropout(0.5))
    base_model.add(Convolution2D(GESTURE_CATEGORIES, (1, 1), padding='valid'))
    base_model.add(Activation('relu'))
    base_model.add(GlobalAveragePooling2D())
    base_model.add(Activation('softmax'))

    return base_model

#map function for the gestures
def label_mapper(val):
    return CATEGORY_MAP[val]

#folder for the images
img_folder = 'Image_collection'


    
#load the images into "input_data"
input_data = []
for sub_folder_name in os.listdir(img_folder):
    path = os.path.join(img_folder, sub_folder_name)
    for fileName in os.listdir(path):
        if fileName.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, fileName))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (225, 225))
            input_data.append([img, sub_folder_name])

# Zip function to separate the 'img_data'(input image) & 'labels' (output text labels) 
img_data, labels = zip(*input_data)

#convert the gesture names to numerical value
labels = list(map(label_mapper, labels))


# performing one hot encoding
labels = np_utils.to_categorical(labels)

# define the model
model = def_model_param()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit model
model.fit(np.array(img_data), np.array(labels), epochs=15)
print("Complete")
# save the trained model
model.save("MRI_gesture_cnn.h5")
