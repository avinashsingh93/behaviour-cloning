import os
import csv
import cv2
import numpy as np
import sklearn

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        next(reader,None)#neglecting the first line as it is titled row
        samples.append(line)

from sklearn.model_selection import train_test_split
#In this section, i splitted my dataset into training and validation dataset with ratio 80:20
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import  sklearn
from random import shuffle
            
def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            
            images=[]
            angles=[]
            for batch_sample  in batch_samples:
                name='./data/IMG/'+batch_sample[0].split('/')[-1]
                center_image=cv2.imread(name)
                center_angle=float(batch_sample[3])
                #appending image from center camera to the list images
                images.append(center_image)
                #appending steering angle from center camera image to list angles
                angles.append(center_angle)
                #Flipping image and angle in order to get more dataset.
                images_fl=np.fliplr(center_image)
                center_angle_fl=-center_angle
                images.append(images_fl)
                angles.append(center_angle_fl)
                #taking correction values as 0.23 for both left and right images
                name='./data/IMG/'+batch_sample[1].split('/')[-1]
                left_image=cv2.imread(name)
				#adding correction value to left steernig angle
                left_angle=center_angle+0.23
                #appending image from left camera to the list images
                images.append(left_image)
                #appending steering angle from left camera image to list angles
                angles.append(left_angle)
                #Flipping image and angle in order to get more dataset.
                image_l_fl=np.fliplr(left_image)
                left_angle_fl=-left_angle
                images.append(image_l_fl)
                angles.append(left_angle_fl)
                
                
                name='./data/IMG/'+batch_sample[2].split('/')[-1]
                right_image=cv2.imread(name)
				#subtracting correction value to right steering angle
                right_angle=center_angle-0.23
                #appending image from right camera to the list images
                images.append(right_image)
                #appending steering angle from right camera image to list angles
                angles.append(right_angle)
                #Flipping image and angle in order to get more dataset.
                image_r_fl=np.fliplr(right_image)
                right_angle_fl=-right_angle
                images.append(image_r_fl)
                angles.append(right_angle_fl)
                
            X_train=np.array(images)
            y_train=np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            #yield name
    
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#print((next(train_generator)))

from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.layers import Lambda
from keras.layers import Flatten,Dense,Lambda,Cropping2D,ELU,Dropout
import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
#Cropping the image so as to remove unwanted data from the image
model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
#normalizing the cropped image
model.add(Lambda(lambda x: x/127.5-1.0))
#added concolutional network 5x5 with 24 as depth
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
#added convolutional network 5x5 with 36 as depth
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
#added convolutional network 5x5 with 48 as depth
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
#added convolutional network 3x3 with 64 as depth
model.add(Convolution2D(64, 3, 3, activation='relu'))
#added convolutional network 3x3 with 64 as depth
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
#Used RELU activation layer
model.add(Dense(1000, activation='relu'))
#used dropout in order to avoid overfitting
model.add(Dropout(0.5))
model.add(Dense(100))
#After this again used dropout of 0.4
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#Used adam optimizer technique so that i did not have to tune learningn rate manually
model.compile(loss='mse',optimizer='adam')
#I am taking samples_per_epoch as 6 times length of train samples as i have flipped all the 3 images making my dataset 6 times
model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')
    
                
