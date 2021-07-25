# Deep-Learning-Internship-Project

Hello and welcome to my work on the project for my summer internship for The Smart Bridge.</br>
This is a deep learning project that uses hand gesture prediction to browse through and zoom or rotate through MRI images during surgery without risk of infection. </br>
To know more about why this solution was proposed I advice you to read the project report. </br>
Here is a video recording that specifies how everything works, the video explains the code AND has a demonstration of how the real-time gesture prediction and action worked in the end. If you want to dig deeper into how the code works continue reading. </br>

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/rwNinT-MUtc/0.jpg)](https://www.youtube.com/watch?v=rwNinT-MUtc)


This is an image to summerize how the project is suppose to work along with a block diagram to make everything clearer from the start. </br>
![Test Image ](/software_designs.png)
![Test Image ](/block.JPG)

Now let me explain how the code works!
At first I started with a python script that would help me collect the images for the model. </br>
I ended up taking 1300 images for each gesture beacause the more pictures I added the better the model was. </br>
There are 5 gestures: </br>
Rotate counterclockwise </br>
Rotate clockwise </br>
Scroll </br>
Zoom in </br>
Zoom out </br>
I also added an empty image so when there is no gesture it does nothing. </br>
All the hand gestures are in the report AND as well as the video. </br>
After capturing all the images I trained my model. </br>
First I gave each gesture a numerical value: </br>

```python
CATEGORY_MAP = {
    "scroll" : 0,
    "zoomin" : 1,
    "zoomout" : 2,
    "rotatecw" : 3,
    "rotateccw" : 4,
    "empty" : 5
}
```
After importing the images and labeling them as well as performing one hot encoding on them, I compiled the model, fit it then saved it. </br>
```python
model = def_model_param()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit model
model.fit(np.array(img_data), np.array(labels), epochs=15)

# save the trained model
model.save("MRI_gesture_cnn.h5")
```
In the next step I tested the model and after that used it for to create the gesture-based tool to capture and predict hand gesture and perform actions on MRI images depending on the gesture. </br>

First I needed a video capture and of course I used cv2, python's very own computer vision library also called OpenCV-python. </br>

```python
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
```
I drew a rectangle in the video frame to better capture the hand gesture and for that you need to view the demonstration in the video. </br>
I loaded the model of course and then i used it to predict the gesture.

```python
model = load_model("MRI_gesture_cnn.h5")
prediction = model.predict(np.array([img]))
gesture_numeric = np.argmax(prediction[0])
gesture_name = mapper(gesture_numeric)
```
And then based on the gesture, an action is performed on the MRI image. There are 5 gestures here, is just one of them. </br>

```python
 if gesture_name == 'rotatecw':
        rot = rotate(MRIimg, -50, reshape=False)
        cv2.imshow("MRI image", rot)
```

In the end, everything went smoothly just as we can see in the demonstration video. </br>
Thank you for your time!




