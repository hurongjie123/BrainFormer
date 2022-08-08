import os, sys, cv2, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from models.resnet3D import resnet3D
from utils.dataset import fMRIdataset
from utils.logger import Logger
from utils.evaluation import compute_performance, compute_performance_class


device_id = torch.device('cuda', 0)
###########   Dataset   ###########
val_dataset = fMRIdataset(
	dataset_dir=img_dir, 
	ann_file='list_2class/list_ADNI_val.txt', 
	size=(64, 64, 48))
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, 
	batch_size = 1, shuffle = False, num_workers = 4)

test_dataset = fMRIdataset(
	dataset_dir=img_dir, 
	ann_file='list_2class/list_ADNI_test.txt',  
	size=(64, 64, 48))
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
	batch_size = 8, shuffle = False, num_workers = 4)

###########   MODEL   ###########
model = resnet3D(depth=18, num_classes=class_num, 
	pretrained='../resnet18-5c106cde.pth')
model.to(device_id)
weight = 'weight/BrainModel_009.pth'
weight = torch.load(weight, map_location='cpu')
model.load_state_dict(weight)

print('model load')

criterion = nn.CrossEntropyLoss()

print('start test on val set')
model.eval()
val_loss, preds = 0.0, []
for i, data in enumerate(val_loader, 1):

	with torch.no_grad():

		images, label, idx = data
		images = images.to(device_id)
		label = label.to(device_id)

		out = model(images)[0]
		_, pred = torch.max(out, 1)
		preds.append(pred)

		loss = criterion(out, label)
		val_loss += loss.item() * label.size(0)

val_loss = val_loss/len(val_dataset)
preds = torch.cat(preds).cpu()
labels = torch.tensor(val_dataset.label_list)

accuracy, precisions, recall, F1_score = compute_performance(preds, labels, class_num=class_num)

print('Val loss: %.3f, Acc: %.3f, precision: %.3f, recall: %.3f, F1_score: %.3f'%(val_loss, accuracy, precisions, recall, F1_score))
print('Finish {} epoch'.format(epoch+1))
torch.save(model.state_dict(), 'weight/Vit_medical_%03d.pth'%(epoch))
#break

print('start test on test set')
model.eval()
test_loss, preds = 0.0, []
for i, data in enumerate(test_loader, 1):

	with torch.no_grad():

		images, label, idx = data
		images = images.to(device_id)
		label = label.to(device_id)

		out = model(images)[0]
		_, pred = torch.max(out, 1)
		preds.append(pred)

		loss = criterion(out, label)
		test_loss += loss.item() * label.size(0)

test_loss = test_loss/len(test_dataset)
preds = torch.cat(preds).cpu()
labels = torch.tensor(test_dataset.label_list)

accuracy, precisions, recall, F1_score = compute_performance(preds, labels, class_num=class_num)	
print('Test loss: %.3f, Acc: %.3f, precision: %.3f, recall: %.3f, F1_score: %.3f'%(test_loss, accuracy, precisions, recall, F1_score))

overall_acc, precisions, recalls, F1_scores, accs = compute_performance_class(preds, labels, class_num=class_num)	
print('overall_acc:', overall_acc)
print('accs:', accs)
print('precisions:', precisions)
print('recalls:', recalls)
print('F1_scores:', F1_scores)