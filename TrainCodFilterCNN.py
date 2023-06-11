# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:48:05 2023

@author: Alfonso Blanco
"""

# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

dirname="Training"
dirnameTest="Test"
batch_size = 16
epochs = 500
######################################################################
import tensorflow as tf
from tensorflow import keras

import numpy as np

import cv2


import os
import re

import imutils

#####################################################################

def loadCodFilterTraining(dirname):
    thresoldpath = dirname 
    
    
    arry=[]
   
   
    print("Reading codfilters from ",thresoldpath)
        
    Conta=0
    
    for root, dirnames, filenames in os.walk(thresoldpath):
        
       
        for filename in filenames:
           
            
            if re.search("\.(txt)$", filename):
                Conta=Conta+1
                #arry=[]
               
                filepath = os.path.join(root, filename)
              
              
                f=open(filepath,"r")
               
               
                for linea in f:
                    
                    
                    
                    arry.append(int(linea))
                        
              
                f.close() 
               
                  
                
    
    
   
    Y_train=np.array(arry)
    
    return  Y_train
   


#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname 
    
    images = []
    imagesFlat=[]
    Licenses=[]
    
    Conta=0
  
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    
    for root, dirnames, filenames in os.walk(imgpath):
        
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                Conta=Conta+1
                
                filepath = os.path.join(root, filename)
                License=filename[:len(filename)-4]
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #gray = cv2.resize(gray, (416,416), interpolation = cv2.INTER_AREA) 
                gray = cv2.resize(gray, (104,104), interpolation = cv2.INTER_AREA) 
                images.append(gray)
                imagesFlat.append(gray.flatten())
                Licenses.append(License)
                         
    return images, Licenses


###########################################################
# MAIN
##########################################################
Y_train=loadCodFilterTraining(dirname)
#print(Y_train)
X_train, Licenses=loadimages(dirname)
X_test, LicensesTest=loadimages(dirnameTest)


X_train=np.array(X_train)
Y_train=np.array(Y_train)

X_test=np.array(X_test)


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# Model / data parameters

# Zero Initialization

# https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
from tensorflow.keras import initializers

initializer = tf.keras.initializers.Zeros()


#input_shape = (416, 416, 1)
input_shape = (104, 104, 1)

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
# Make sure images have shape (140, 140, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

num_classes=11
# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        
        #https://medium.com/imagescv/all-about-pooling-layers-for-convolutional-neural-networks-cnn-c4bca1c35e31
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Flatten(),
        
        #
        #https://medium.com/imagescv/all-about-pooling-layers-for-convolutional-neural-networks-cnn-c4bca1c35e31
        #
        # https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
        layers.Dense(25, activation='relu', kernel_initializer='he_uniform'),
        #layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax",kernel_initializer=initializer),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

#test_scores = model.evaluate(x_test, y_test, verbose=2)
#print("Test loss:", test_scores[0])
#print("Test accuracy:", test_scores[1])
#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])


predictions = model.predict(X_test)

predictions=np.argmax(predictions, axis=1)


print("Number of imagenes to test : " + str(len(X_train)))
print("Number of  CodFilters  : " + str(len(Y_train)))



# https://www.kaggle.com/getting-started/145864
# serialize model to JSON
model_json = model.to_json()
with open("FilterCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("FilterCNN.h5")
TotHits=0
TotFailures=0

NumberImageOrder=0

for i in range (len( LicensesTest)):
    
   
   
    
    NumberImageOrder=NumberImageOrder+1
    CodFilter=predictions[i]
    
    print(LicensesTest[i] + "  CodFilter = "+ str(CodFilter))
    
    
    