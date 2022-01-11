from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
print(tf.__version__)





from tensorflow.keras.optimizers import Adam



import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="C:/PythonKodlar/YSAPROJE/NewData/Train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="C:/PythonKodlar/YSAPROJE/NewData/Test", target_size=(224,224))

batch_size=32

print(len(traindata))

SPE = len(traindata)//batch_size
print(SPE)
VS = len(testdata)//batch_size


model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()


print("TEST")

Epochs=20
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=SPE,generator=traindata, validation_data= testdata, validation_steps=10,epochs=Epochs,callbacks=[checkpoint,early])

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


try:
    keras.models.save_model(model,'C:/PythonKodlar/YSAPROJE/Vgg16Ok')
    print("Kaydedildi")
except:
  print("Hata Oldu")

 
    

def array2dir(array):
    if array[0][0] > array[0][1] and array[0][0] > array[0][2]:
            print("Square")

    if array[0][1] > array[0][0] and array[0][1] > array[0][2]:
            print("Star")
    if array[0][2] > array[0][0] and array[0][2] > array[0][1]:
            print("Arrow")




import time

from keras.preprocessing import image

 
#model.save('C:/PythonKodlar/YSAPROJE/StarAndSquare/Model')
#model = keras.models.load_model('C:/PythonKodlar/YSAPROJE/StarAndSquare/Model')


def take_photo():
    vid = cv2.VideoCapture(0)
  
    while(True):
          
    
        
        ret, image = vid.read()
        # Display the resulting frame
        cv2.imshow('Kamera', image)

        image=np.array(image)

        image = image.reshape((1, 480, 640, 3))

   
        #img = image.load_img(image,target_size=(224,224))
        #img = image.load_img("image.jpeg",target_size=(224,224))
        img = np.asarray(image)
        plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        from keras.models import load_model
        saved_model = load_model("vgg16_1.h5")
        output = saved_model.predict(img)
        if output[0][0] > output[0][1]:
            print("cat")
        else:
            print('dog')
            
        time.sleep(0.5)    
        
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
print("Realtimeöncesi")
#take_photo()
print("RealtimeSonrası")




