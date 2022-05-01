import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
from cv2 import imread,imwrite,GaussianBlur
from pydensecrf.utils import unary_from_labels,unary_from_softmax,create_pairwise_bilateral,create_pairwise_gaussian
'''
%matplotlib inline

plt.rcParams['image.interpolation'] = 'nearsest'
plt.rcParams['image.cmap'] = 'gray'

FOLDER_PATH = "home/euclid/ywang440"

input: image path+name, annotation path+name, output name
output: numpy array that contains the inferenced image. max(pixels) = 1
'''
#img, anno are (batchsize,w,h)
def ppsoftmax(img, anno,times = 70,gt_prob=0.05,blur=5,sdims=(3,3),schan=(0.04,),bcompat=4,gcompat=3,sxy=(3,3)):

    # add unary energy, flattens it.
    #first class in anno is #classes. shape (2,h,w), (0,:,:) are foreground, (1,:,:) are background
    #anno2 = np.expand_dims(anno,axis=0)
    temp = np.zeros((2,anno.shape[1],anno.shape[2]))
    temp[0,:,:] = anno[0,:,:]
    temp[1,:,:] = 1 - anno[0,:,:]
    U = unary_from_softmax(temp,scale=gt_prob)

    d = dcrf.DenseCRF2D(anno.shape[1],anno.shape[2],2)

    d.setUnaryEnergy(U)

    #addPairwise energy
    pairwise_energy = create_pairwise_bilateral(sdims=sdims,schan=schan,img=img, chdim=0)
    d.addPairwiseEnergy(pairwise_energy, compat=bcompat)  # `compat` is the "strength" of this potential.
    d.addPairwiseGaussian(sxy=sxy,compat=gcompat,kernel = dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    #Do Inference
    Q = d.inference(times)

    MAP = np.argmax(Q,axis = 0)
    #print(U.shape - np.sum(MAP))
    return (1-MAP).reshape((anno.shape[1],anno.shape[2]))



def postprocessing(img, anno,times = 70):

    # add unary energy, flattens it.
    colors,labels = np.unique(anno,return_inverse=True)
    #print(colors) #print(labels.shape)
    U = unary_from_labels(labels,2,gt_prob=0.05,zero_unsure=False)

    d = dcrf.DenseCRF2D(anno.shape[0],anno.shape[1],2)
    d.setUnaryEnergy(U)

    img2 = GaussianBlur(img,(3,3),3)
    img2 = np.expand_dims(img,axis=2)

    #addPairwise energy
    pairwise_energy = create_pairwise_bilateral(sdims=(3,3),schan=(0.04,),img=img2, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=3)  # `compat` is the "strength" of this potential.
    d.addPairwiseGaussian(sxy=(3,3),compat=4,kernel = dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    #Do Inference
    Q = d.inference(times)

    MAP = np.argmax(Q,axis = 0)
    print(labels.shape - np.sum(MAP))
    return MAP.reshape((anno.shape[0],anno.shape[1]))
'''
input shape width, depth, height
'''
def postprocessing3d(img, anno, times = 10):
	colors, labels = np.unique(anno,return_inverse=True)
	U = unary_from_labels(labels,2,gt_prob=0.54,zero_unsure=False)
	d = dcrf.DenseCRF(np.prod(img.shape),2)
	d.setUnaryEnergy(U)

	feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=img.shape)
	d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

	pairwised = create_pairwise_bilateral(sdims=(1,1,1),schan=(0.01),img=img, chdim=-1)
	d.addPairwiseEnergy(pairwised,compat=5, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	#stay away from smoothing so far
    	#img2 = GaussianBlur(img,(5,5),5)

	Q = d.inference(times)
	MAP = np.argmax(Q, axis=0).reshape((img.shape))
	print(labels.shape - np.sum(MAP))
	return MAP 


