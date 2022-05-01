import sys
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
from cv2 import imread,imwrite
from pydensecrf.utils import unary_from_labels,create_pairwise_bilateral,create_pairwise_gaussian
from postprocessing import postprocessing,postprocessing3d,ppsoftmax
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    getpred_loader,
    check_accuracy,
    save_predictions_as_imgs,
    randomSample,
    check_accuracy_RS
)

'''
This file performs merging over the x-y prediction and y-z prediction

'''
FOLDER_PATH = "/home/euclid/ywang440/test1/"
INPUT = FOLDER_PATH + sys.argv[1] #folders that contains the predicted images
ANNO = FOLDER_PATH + sys.argv[2] #folders that contain the annotations
OUTPUT = FOLDER_PATH + sys.argv[3] #folders to store the output predictions, according to the image name


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 1
IMAGE_HEIGHT = 512  # 512
IMAGE_WIDTH = 512  # 512
PIN_MEMORY = True
CLASS_WEIGHT = torch.Tensor([0.3,0.7]) #default 1,1,1 affects crossentropyloss

#LISTOFGOODPHYPERS
#For y-z:
gt_prob1 = 0.585
times1 = 40
blur1 = 3
bcompat1 = 7
gcompat1 = 3
sdims1 = (2,2)
sxy1 = (4,4)
schan1 = (0.074,) 

#For x-y:
gt_prob = 0.669
times = 40
blur = 6
bcompat = 2
gcompat = 2
sdims = (2,2)
sxy = (4,4)
schan = (0.109,) 

def main():

	transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
 		),
            ToTensorV2(),],)
        
	resize = A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
	resizeNorm = A.Compose([resize,A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
 		)])
	#Load U-Net Models:          
	model = UNET(in_channels=3, out_channels=1).to(DEVICE)
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	scaler = torch.cuda.amp.GradScaler()
	load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

	#Predict a sequence of images, stack them in order:
	stackimg = list()
	stackanno = list()
	stackpred = list()
	names = []
	with torch.no_grad():
		for f in os.listdir(INPUT):
			raw = INPUT + f
			anno = ANNO + f
			names.append(OUTPUT + f)
			#predictions
			x = transform(image=imread(raw))['image'] #reads as RGB to fit U_NET
			x = x.unsqueeze(0).to(DEVICE)	#expand one dimension
			preds = torch.sigmoid(model(x))
			#preds = (preds > 0.5).float() 
			#stack the raw images and predictions 512,512
			stackimg.append(resize(image=imread(raw,0))['image']) # 0 as to read in gray_scale mode
			stackpred.append(preds.to("cpu").numpy()[0,0,:,:]) #reverse height,width to width,height
			stackanno.append(resizeNorm(image=imread(anno,0))['image'])

	#Predicted images and annotations 
	images = np.stack(stackimg,axis=2) #(x,y,z)
	predictions = np.stack(stackpred,axis=2) #(x,y,z)
	annotations = np.stack(stackanno,axis=2) #(x,y,z)

	#do y-z direction
	MAPs = np.zeros(images.shape)
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	for i in range(images.shape[0]):
		MAP =  ppsoftmax(np.expand_dims(images[i,:,:],0),
			np.expand_dims(predictions[i,:,:],0),
			times=times1,gt_prob=gt_prob1,
			blur=blur1,sdims=sdims1,
			schan=schan1,
			bcompat=bcompat1,
			gcompat=gcompat1,sxy=sxy1)
		#calculate Dice Score accumulatively.
		MAPs[i,:,:] = MAP	
		y = annotations[i,:,:]
		num_correct += (MAP == y).sum()
		num_pixels += np.prod(MAP.shape)
		dice_score += (2 * (MAP * y).sum()) / ((MAP + y).sum() + 1e-8)
	print(f"Dice score: {dice_score/images.shape[0]}")

	#do x-y direction
	MAP2s = np.zeros(images.shape)
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	for i in range(images.shape[2]):
		MAP =  ppsoftmax(np.expand_dims(images[:,:,i],0),
			np.expand_dims(predictions[:,:,i],0),
			times=times,gt_prob=gt_prob,
			blur=blur,sdims=sdims,
			schan=schan,
			bcompat=bcompat,
			gcompat=gcompat,sxy=sxy)
		#calculate Dice Score accumulatively.
		MAP2s[:,:,i] = MAP	
		y = annotations[:,:,i]
		num_correct += (MAP == y).sum()
		num_pixels += np.prod(MAP.shape)
		dice_score += (2 * (MAP * y).sum()) / ((MAP + y).sum() + 1e-8)
	print(f"Dice score: {dice_score/images.shape[2]}")
	
	#logical_or, merge x-y and y-z:
	Res = np.logical_or(MAPs,MAP2s).astype(int)

        #calculate Dice score:
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	for i in range(images.shape[2]):
		#calculate Dice Score accumulatively.
		MAP = Res[:,:,i] 
		y = annotations[:,:,i]
		num_correct += (MAP == y).sum()
		num_pixels += np.prod(MAP.shape)
		dice_score += (2 * (MAP * y).sum()) / ((MAP + y).sum() + 1e-8)
	print(f"Dice score: {dice_score/images.shape[2]}")

	#write image as x-y planes:
	for i in range(images.shape[2]):
		imwrite(names[i],Res[:,:,i]*255)


if __name__ == "__main__":
	main()

