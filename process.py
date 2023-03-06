import numpy as np
import cv2
import os

root = 'data/' # change to your data folder path
data_f = ['ISIC2018_Task1-2_Training_Input/', 'ISIC2018_Task1-2_Validation_Input/', 'ISIC2018_Task1_Testing_Input/']
mask_f = ['ISIC2018_Task1_Training_GroundTruth/', 'ISIC2018_Task1_Validation_GroundTruth/', 'ISIC2018_Task1_Testing_GroundTruth/']

set_size = [2074,520]
save_name = ['train','test']

height = 352
width = 352

for j in range(2):

	print('processing ' + data_f[j] + '......')
	count = 0
	length = set_size[j]
	imgs = np.uint8(np.zeros([length, height, width, 3]))
	masks = np.uint8(np.zeros([length, height, width]))

	path = root + data_f[j]
	mask_p = root + mask_f[j]

	for i in os.listdir(path):
				img = cv2.imread(path+i)
				# BGR → RGB
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (width, height))
				#m_path = mask_p + i.replace('.tif', '.xml')
				m_path = mask_p + i

				mask = cv2.imread(m_path, 0)
				mask = cv2.resize(mask, (width, height))
				imgs[count] = img
				masks[count] = mask
				count +=1
				print(count)
	np.save('{}/data_{}.npy'.format(root, save_name[j]), imgs)
	np.save('{}/mask_{}.npy'.format(root, save_name[j]), masks)
