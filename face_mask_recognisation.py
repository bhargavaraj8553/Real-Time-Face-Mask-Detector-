import numpy as np
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.models import Sequential, load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import datetime
import cv2

# training the model and make it to recognise the mask and non - mask images using a pre-determined dataset
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D())
# the first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu'))

# At last using the sigmoid activation function so that output will be between 0 and 1 .
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Normalising image data for training purpose
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Normalising image data for testing purpose
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

# Training the model and validating it against the testing data
model_saved=model.fit_generator(
        training_set,
        epochs=10,
        validation_data=test_set,
        )

model.save('mymodel.h5',model_saved)

# Testing the model and comparing the predicted value and the desired value
obtained_model = load_model('mymodel.h5')
prediction = obtained_model.predict(test_set)
print(prediction[97])
print(test_set.classes[97])


# Testing the model and predicting the value of one of the new image
obtained_model = load_model('mymodel.h5')

# test_imagee = cv2.imread("test\with_mask\85-with-mask.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("85-with-mask", test_imagee)
# cv2.waitKey(0)

check_img = load_img(r"test\with_mask\85-with-mask.jpg", target_size=(150, 150, 3))
check_img=img_to_array(check_img)
check_img=np.expand_dims(check_img, axis=0)
obtained_model.predict(check_img)[0][0]
if (obtained_model.predict(check_img)[0][0]==1): print ('person is not wearing mask')
else: print('person is wearing mask')


# Finally taking the live input from camera and testing the model against that input
obtained_model = load_model('mymodel.h5')

# Note- Here VideoCapture(0) will take input from inbuilt webcam , in case external webcam is used it can be changed accordingly
webcam_inp = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while webcam_inp.isOpened():
    _,im=webcam_inp.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=3)
    for(x,y,w,h) in face:
        face_img = im[y:y+w, x:x+w]

        cv2.imwrite('temp.jpg',face_img)
        check_img=load_img('temp.jpg', target_size=(150, 150, 3))
        check_img=img_to_array(check_img)
        check_img=np.expand_dims(check_img, axis=0)
        val_obtained=obtained_model.predict(check_img)[0][0]

        if val_obtained==1:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(im,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(im,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        curr_date=str(datetime.datetime.now())
        cv2.putText(im, curr_date, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('testing image',im)

    if cv2.waitKey(1)==ord('q'):
        break

webcam_inp.release()
cv2.destroyAllWindows()

