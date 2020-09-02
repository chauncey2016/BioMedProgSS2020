from torch.utils.data import Dataset
import cv2
import numpy as np

class BioMedDataset(Dataset):
	"""Dataset for the bio med practicum"""

	def __init__(self, image_files, labels, transforms):
		self.image_files = image_files
		self.labels = labels
		self.transforms = transforms

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, index):
		return self.transforms(self.image_files[index]), self.labels[index]

class LoadToRGB(object):
	"""Load image and convert grayscale to rgb."""

	def __call__(self, sample):
		image = cv2.imread(sample, cv2.IMREAD_COLOR)
		image = np.transpose(image, (2, 0, 1))
		#print(image.shape, image.ndim)
		return image

class LoadToGrayscale(object):
	"""Load image and convert to grayscale."""

	def __call__(self, sample):
		image = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
		#print(image.shape, image.ndim)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		#gray = sample
		#if sample.ndim > 2:
			#print("ndim = ", sample.ndim, sample.shape)
			#print("before conversion:", sample)
		#	gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
			#print("converted:", gray.ndim, gray.shape, gray)

			#pass
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		#print(image.shape, image.ndim)
		#image = image.transpose((2, 0, 1))
		#print(image.shape, image.ndim)
		return image

class Shape(object):
	"""Print data shape for debugging"""

	def __call__(self, sample):
		print(sample.shape)
		return sample

class EmptyTransform(object):
	"""Don't do anything"""

	def __call__(self, sample):
		return sample

class PrintToFileAndConsole:
	"""Print every message to log file as well as to console"""
	def write(self, *args, **kwargs):
		self.out1.write(*args, **kwargs)
		self.out2.write(*args, **kwargs)

	def flush(self):
		self.out1.flush()
		self.out2.flush()
		
	def __init__(self, out1, out2):
		self.out1 = out1
		self.out2 = out2