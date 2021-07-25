import cv2
from keras.models import load_model
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom


#initializing attriutes
gesture_name=""
click = False
i = 1
font = cv2.FONT_HERSHEY_PLAIN
filepath = r'C:\Users\kille\Desktop\MRIimages\\'
#the gestures that we have
CATEGORY_MAP = {
    0 : "scroll",
    1 : "zoomin",
    2 : "zoomout",
    3 : "rotatecw",
    4 : "rotateccw",
    5 : "empty"
}

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
#we have to load the module
model = load_model("MRI_gesture_cnn.h5")
#map function for the gestures
def mapper(val):
    return CATEGORY_MAP[val]
#zoom function
def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return out



while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    #video settings
    cv2.rectangle(image, (200, 200), (550, 550), (255, 255, 255), 2)
    cv2.putText(image, "Fit the gesture inside the white box and Press 's' ",(20, 30), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Press 'q' to exit.", (20, 60), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "prediction: {}".format(gesture_name),(20, 100), font, 1, (12, 20, 200), 2, cv2.LINE_AA)
    cv2.imshow("Video Frame", image)
    #MRI image
    imgpath = r'C:\Users\kille\Desktop\MRIimages\1.jpg'
    MRIimg = cv2.imread(imgpath)
    MRIimg = cv2.resize(MRIimg, (500, 700))
    cv2.imshow("MRI image", MRIimg)


    if not ret:
        continue

    #start prediction when click on 's'
    if click:
        region_of_interest = image[200:550, 200:550]
        img = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB)
        img = cv2.resize(region_of_interest, (225, 225))
        prediction = model.predict(np.array([img]))
        gesture_numeric = np.argmax(prediction[0])
        gesture_name = mapper(gesture_numeric)
    #the actions
    if gesture_name == 'rotatecw':
        rot = rotate(MRIimg, -50, reshape=False)
        cv2.imshow("MRI image", rot)

    if gesture_name == 'rotateccw':
        rot = rotate(MRIimg, 50, reshape=False)
        cv2.imshow("MRI image", rot)

    if gesture_name == 'scroll':
        i = i + 1
        newimgpath = filepath + str(i) + '.jpg'
        print(newimgpath)
        newimg = cv2.imread(newimgpath)
        newimg = cv2.resize(newimg, (500, 700))
        cv2.destroyWindow("MRI image")
        cv2.waitKey(1)
        cv2.imshow("next MRI image", newimg)
        cv2.waitKey(500)
        click = not click
    if gesture_name == 'zoomin':
        zm1 = clipped_zoom(MRIimg, 1.5)
        cv2.imshow("MRI image", zm1)

    if gesture_name == 'zoomout':
        zm2 = clipped_zoom(MRIimg, 0.5)
        cv2.imshow("MRI image", zm2)

    if gesture_name == 'empty':
        cv2.waitKey(1000)

    k = cv2.waitKey(10)
    if k==ord('q'):
            break
    if k == ord('s'):
        click = not click



print(gesture_name)
video.release()
cv2.destroyAllWindows()
