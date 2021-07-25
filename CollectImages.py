import cv2
import os
import sys

try:
    num = int(sys.argv[1])
    img_label = sys.argv[2]
except:
    print("\nInvalid syntax.")
    exit(-1)

font = cv2.FONT_HERSHEY_PLAIN
click = False
#the folders
img_folder = 'Image_collection'
label_name = os.path.join(img_folder, img_label)

#count of images
count = image_name = 0

try:
    os.mkdir(img_folder)
except FileExistsError:
    pass
try:
    os.mkdir(label_name)
except FileExistsError:
    #override image if already exist
    image_name=len(os.listdir(label_name))

    
#open camera
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    
    if not ret:
        continue
    if count == num:
        break


    cv2.rectangle(image, (200, 200), (550, 550), (255, 255, 255), 2)
    if click:
        region_of_interest = image[200:550, 200:550]
        save_path = os.path.join(label_name, '{}.jpg'.format(image_name + 1))
        cv2.imwrite(save_path, region_of_interest)
        image_name += 1
        count += 1
    cv2.putText(image, "Fit your hand in the rectangle and press s",(20, 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Press 'q' to exit.",(20, 60), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Image Count: {}".format(count),(20, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Collect images", image)

    k = cv2.waitKey(10)
    if k==ord('q'):
            break
    if k == ord('s'):
        click = not click

print("\n\nDone\n\n")
video.release()
cv2.destroyAllWindows()
