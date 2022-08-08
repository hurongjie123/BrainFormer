import os, torch, cv2, time, sys
import torch.nn.functional as F
import numpy as np
#import resnet


def VisulizationMap(imgs, mask):

	#print(imgs.shape, mask.shape)
	imgs = F.relu(imgs)
	imgs = (imgs-imgs.min())/(imgs.max()-imgs.min())*255
	imgs = imgs[0,0]

	imgs = imgs.view(imgs.size(0), imgs.size(1), 8, 8)
	imgs = imgs.permute(2,0,3,1)
	imgs = imgs.reshape(imgs.size(0)*imgs.size(1), imgs.size(2)*imgs.size(3))
	imgs = imgs.cpu().numpy()
	imgs = imgs.astype(np.uint8)


	#mask = F.interpolate(mask, size=(64,72,64), mode='trilinear', align_corners=False)

	mask = mask[0,0]*255
	mask = mask.view(mask.size(0), mask.size(1), 8, 8)
	mask = mask.permute(2,0,3,1)
	mask = mask.reshape(mask.size(0)*mask.size(1), mask.size(2)*mask.size(3))
	mask = mask.cpu().numpy()
	mask = mask.astype(np.uint8)

	
	draw = np.concatenate([imgs, mask], axis=1)
	return draw
	
	#cv2.imwrite('draw/%04d.jpg'%(i), draw)
	#aaa

