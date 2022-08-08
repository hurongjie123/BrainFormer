import os, sys, cv2, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import torch.nn as nn

from models.resnet3D import resnet3D
from utils.dataset import fMRIdataset, fMRIdataset_seq
from utils.Visulization import VisulizationMap
from utils.gradcam import GradCAM, GradCAMpp

device_id = torch.device('cuda', 0)
##########   DATASET   ###########
img_dir = '/home/ljn/disk1/medic_data/data09.25_data2/FunImgTARWS/'
val_dataset = dataset.fMRIdataset_seq(
	dataset_dir=img_dir, 
	ann_file='list/seq_list_FunImgTARWS_test0.txt', 
	size=(64, 72, 64))
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, 
	batch_size = 1, shuffle = False, num_workers = 0)

###########   MODEL   ###########
model = resnet3D(depth=18, num_classes=3, 
	pretrained='resnet18-5c106cde.pth')
model.to(device_id)
print('model load')
weight = 'weight/Vit_medical_000.pth'
weight = torch.load(weight, map_location='cpu')
model.load_state_dict(weight)

resnet_model_dict = dict(type='resnet3d', arch=model.eval(), layer_name='layer2', input_size=(64, 72, 64))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
###########   TRAIN   ###########


print('start test on val set')
#model.eval()
val_loss, preds = 0.0, []
for ind, data in enumerate(val_loader, 1):
	print()

	images, label, idx = data
	img_name = val_dataset.img_list[idx].split('/')[-1]
	#images = images.to(device_id)
	label = label.to(device_id)
	images = images[0]
	print(images.size(), label)

	masks = []
	for i in range(images.size(0)):

		image = images[i].unsqueeze(dim=0)
		image = image.to(device_id)
		print(i, image.size(), label)

		mask, _ = resnet_gradcam(image, class_idx=label.item())
		#draw(images, mask)
		masks.append(mask)
		#print('mask', mask.size(), label)
		
	masks = torch.cat(masks, dim=0)
	print('masks', masks.size())
	masks = masks.mean(dim=0, keepdim=True)
	print('masks2', masks.size())
	draw = VisulizationMap(images, masks)

	cv2.imshow('1', draw)
	cv2.waitKey(0)
	cv2.imwrite('draw/%s.jpg'%(img_name), draw)
