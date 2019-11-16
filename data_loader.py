"""

Script used to load in the data for predicting the speed of a car in a given frame. 

Uses the difference between two images to (ideally) identify what sorts of differences between the two
frames correspond to a larger difference, which in turn means the speed of the car has changed drastically. 


"""
#import dependencies

import numpy as np
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


#designated batch size for model training. 
def load_trainval_data(batch_size = 32):
	"""
	This function iterates through the video and creates a subtracts the current frame from the previous
	one, and uses this as the input into a model who's output is the given speed. 

	Currently only designed to load in just training/validation data, and once I get a good enough result,
	will try on the testing file!

	Parameters
	-----------
	batch_size : int
		Designated # of difference images to plug into the model at one training step.

	Returns
	-------

	train_loader : list
		List of training batches.
	val_loader : list
		List of validation batches. 

	"""

	#iterate over the image at the designated frame rate, and append a list with the difference btw
	#that frame and the previous one. Definitely not the best approach but there is room to improve it!

	vidcap = cv2.VideoCapture('data/train.mp4')
	train_speeds = np.loadtxt('data/train.txt')
	sec = 0
	frameRate = 1/20 #//it will capture image in each 0.5 second
	count=0
	imgs = []
	img_diff = []
	hasFrames = True
	while hasFrames:
	 #    print(count)
	    count = count + 1
	    sec = sec + frameRate
	   # sec = round(sec, 2)
	  #  vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	    hasFrames,image = vidcap.read()
	    imgs.append(image)
	    if count > 1:
	        img_diff.append(imgs[count-1] - imgs[count - 2])


	#now make an array for this, so it can easiy be made into a loader file which in turn 
	#gets converted into a Tensor dtype
	img_diff = np.array(img_diff)
	train_speeds = train_speeds[1:len(img_diff)+1]


	X_train,X_val,y_train,y_val = train_test_split(img_diff,train_speeds,shuffle = False)

	train_loader = []
	x = 0
	while x < len(X_train) // batch_size:
	    Xtrain_batch = []
	    ytrain_batch = []
	    i = 0
	    while i < batch_size:
	        Xtrain_batch.append(X_train[i])
	        ytrain_batch.append(y_train[i])
	        i +=1
	        
	    train_loader.append((torch.Tensor(Xtrain_batch).resize(batch_size,3,480,640),torch.Tensor(ytrain_batch)))
	    X_train = X_train[i:]
	    y_train = y_train[i:]
	    x+=1


	val_loader = []
	x = 0
	while x < len(X_val) // batch_size:
	    Xval_batch = []
	    yval_batch = []
	    i = 0
	    while i < batch_size:
	        Xval_batch.append(X_val[i])
	        yval_batch.append(y_val[i])
	        i +=1
	        
	    val_loader.append((torch.Tensor(Xval_batch).resize(batch_size,3,480,640),torch.Tensor(yval_batch)))
	    X_val = X_val[i:]
	    y_val = y_val[i:]
	    x+=1


	return train_loader,val_loader

