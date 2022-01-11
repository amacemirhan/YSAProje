import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

img = image.load_img("C:/PythonKodlar/YSAPROJE/StarAndSquare/SQSTData/NewData/Train/Arrow/s11 (13).jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

from keras.models import load_model
model=load_model("C:/PythonKodlar/YSAPROJE/StarAndSquare/VggModel")
output = model.predict(img)

print("ArrayTest")

def array2dir(array):
    if array[0][0] > array[0][1] and array[0][0] > array[0][2]:
            print("sol")

    elif array[0][1] > array[0][0] and array[0][1] > array[0][2]:
            print("sağ")

    elif array[0][2] > array[0][1] and array[0][2] > array[0][0]:
            print("yukarı")
            
    else:
            print("HATA!")

array2dir(output)


import cv2
import time

cam = cv2.VideoCapture(0)   # 0 -> index of camera

print("Realtime Start")
while(True):
    s, img = cam.read()
    if s:    # frame captured without any errors
        cv2.imshow('Kamera', img)
        img=cv2.resize(img,(224,224))
        img = np.asarray(img)
        plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        output = model.predict(img)
        print(output)
        array2dir(output)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
 # Destroy all the windows
cv2.destroyAllWindows()