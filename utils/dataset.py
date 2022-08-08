import os, torch, random, cv2, glob
import numpy as np
from torch.utils import data

def volume_pad(volume, pad_shape):

	d, h, w = volume.shape
	d = min(d, pad_shape[0])
	h = min(h, pad_shape[1])
	w = min(w, pad_shape[2])
	volume = volume[0:d, 0:h, 0:w]

	volume_pad = np.zero(pad_shape)
	volume_pad[0:d, 0:h, 0:w] =  volume
	return volume_pad

class fMRIdataset(data.Dataset):
	def __init__(self, dataset_dir, ann_file, size):

		self.dataset_dir = dataset_dir
		self.volume_size = size

		with open(ann_file) as f:
			line = f.readlines()
			#self.img_names = [i.split()[0] for i in line]
			self.img_list = [dataset_dir+i.split()[0] for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):

		im_path = self.img_list[index]
		label = self.label_list[index]
		volume = np.load(im_path) #d*h*w

		volume = volume_pad(volume, self.volume_size)
		volume = np.expand_dims(volume, 0)

		return volume, label, index

	def __len__(self):
		return len(self.label_list)



class fMRIdataset_stride(data.Dataset):
	def __init__(self, dataset_dir, ann_file, size, sample_stride=1):

		self.dataset_dir = dataset_dir
		self.volume_size = size
		self.sample_stride = sample_stride

		with open(ann_file) as f:
			line = f.readlines()
			#self.img_names = [i.split()[0] for i in line]
			self.img_list = [dataset_dir+i.split()[0] for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):

		index = index*self.sample_stride + random.randint(0,self.sample_stride-1)
		im_path = self.img_list[index]
		label = self.label_list[index]

		volume = np.load(im_path) #d*h*w
		volume = volume_pad(volume, self.volume_size)
		volume = np.expand_dims(volume, 0)

		return volume, label, index

	def __len__(self):
		return len(self.label_list)//self.sample_stride


class fMRIdataset_seq(data.Dataset):
	def __init__(self, dataset_dir, ann_file, size):

		self.dataset_dir = dataset_dir
		self.volume_size = size

		with open(ann_file) as f:
			line = f.readlines()
			#self.img_names = [i.split()[0] for i in line]
			self.img_list = [dataset_dir+i.split()[0] for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):

		im_path_prefix = self.img_list[index]
		label = self.label_list[index]

		im_paths = glob.glob(im_path_prefix+'*')

		volumes = []
		for im_path in im_path:

			volume = np.load(im_path) #d*h*w
			volume = volume_pad(volume, self.volume_size)
			volume = np.expand_dims(volume, 0)
			volumes.append(torch.tensor(volume))

		volumes = torch.stack(volumes, dim=0)

		return volumes, label, index

	def __len__(self):
		return len(self.label_list)