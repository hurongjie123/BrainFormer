import torch
import numpy as np

def compute_performance_class(preds, labels, class_num=3):
	print(preds.size(), labels.size())

	recalls, precisions = [], []
	for i in range(class_num):

		label_i = (labels==i).float()
		pred_i = (preds==i).float()
		TP = (label_i*pred_i).sum()

		recall = TP/label_i.sum()
		precision = TP/pred_i.sum() if pred_i.sum()>0 else 0

		recalls.append(recall)
		precisions.append(precision)

	#recalls = torch.tensor(recalls).mean()
	#precisions = torch.tensor(precisions).mean()

	F1_scores = []
	for pre, recall in zip(precisions, recalls):
		if pre==0 or recall==0:
			F1_score = 0
		else:
			F1_score = 1/(1/2*(1/pre+1/recall))
		F1_scores.append(F1_score)

	accs = []
	for i in range(class_num):

		label_i = (labels==i).float()
		pred_i = (preds==i).float()
		TP = (label_i*pred_i).sum()
		TN = ((1-label_i)*(1-pred_i))
		acc = (TP+TN)/labels.size(0)
		accs.append(acc)

	overall_acc = (preds==labels).float().sum()/len(labels)

	return overall_acc, precisions, recalls, F1_scores, accs


def compute_performance(preds, labels, class_num=3):
	print(preds.size(), labels.size())

	recalls, precisions = [], []
	for i in range(class_num):

		label_i = (labels==i).float()
		pred_i = (preds==i).float()
		TP = (label_i*pred_i).sum()

		recall = TP/label_i.sum()
		precision = TP/pred_i.sum() if pred_i.sum()>0 else 0

		recalls.append(recall)
		precisions.append(precision)

	recalls = torch.tensor(recalls).mean()
	precisions = torch.tensor(precisions).mean()
	F1_score = 1/(1/2*(1/precisions+1/recalls))
	accuracy = (preds==labels).float().sum()/len(labels)

	return accuracy, precisions, recalls, F1_score