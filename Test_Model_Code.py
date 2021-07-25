from keras.models import load_model
import cv2
import numpy as np


filepath = r'C:\Users\kille\Pictures\img2.jpg'

#the gestures
CATEGORY_MAP = {
    0 : "scroll",
    1 : "zoomin",
    2 : "zoomout",
    3 : "rotatecw",
    4 : "rotateccw",
    5 : "empty"
}

#map function for the gestures
def mapper(val):
    return CATEGORY_MAP[val]

#load image and resize it
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (225, 225))
#now we have to load the model
model = load_model("MRI_gesture_cnn.h5")
# Predict the gesture from the input image
prediction = model.predict(np.array([img]))

gesture_numeric = np.argmax(prediction[0])
print("the gesture is " , gesture_numeric)
gesture_name = mapper(gesture_numeric)
print("Predicted Gesture: {}".format(gesture_name))
