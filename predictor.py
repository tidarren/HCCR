import os 
import cv2
import pickle
import numpy as np
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import string
import random
import argparse

def get_code(length):
    return ''.join(random.sample(string.ascii_letters + string.digits, length))


class GlobalWeightedAveragePooling2D(GlobalAveragePooling2D):

	def __init__(self, kernel_initializer='uniform', **kwargs):
		self.kernel_initializer = kernel_initializer
		super(GlobalWeightedAveragePooling2D, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name='W',
								 shape=input_shape[1:],
								 initializer=self.kernel_initializer,
								 trainable=True)
		# print('input_shape:', input_shape)
		super(GlobalWeightedAveragePooling2D, self).build(input_shape)

	def call(self, inputs):
		inputs = inputs*self.W # element-wise multiplication for every entry of input
		if self.data_format == 'channels_last':
			return K.sum(inputs, axis=[1, 2])
		else:
			return K.sum(inputs, axis=[2, 3])


# Hyper parameters
IMG_SHAPE = (96,96)
COLOR_MODE = 'rgb'
CHANNEL = 3 if COLOR_MODE=='rgb' else 1



# load label to char dictionary
with open('label2char_801.pkl', 'rb') as f:
    label2char_801 = pickle.load(f)

with open('label2char_879.pkl', 'rb') as f:
    label2char_879 = pickle.load(f)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-ip","--img_path", type=str, required=True, 
						help="Specify the path of image")
	parser.add_argument("-mp","--model_path", type=str, required=True, 
						help="Specify trained model path")
	args = parser.parse_args()
	
	# img_path = '../training_data/combine/丁/0Ogq_select_22.jpg'
	# img_path = '../training_data/combine/巧/0gVm_select_58.jpg'
	# img_path = '../training_data/combine/健/0qr9_88.jpg'
	# img_path = '../training_data/combine/高/0JXn_96.jpg'
	
	IMG_PATH = args.img_path 
	MODEL_PATH = args.model_path 
	
	if 'combine_similar' in MODEL_PATH:
		label2char = label2char_879
	else:
		label2char = label2char_801


	image = cv2.imread(IMG_PATH) 
	
	# load model
	model = load_model(MODEL_PATH, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

	####### PUT YOUR MODEL INFERENCING CODE HERE #######

	# save to tmp
	tmp_dir = 'tmp'
	tmp_file = get_code(6)+'.jpg'
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	tmp_file_path = os.path.join(tmp_dir,tmp_file)
	cv2.imwrite(tmp_file_path, image)

	# preprocess
	img = load_img(tmp_file_path,target_size=IMG_SHAPE, color_mode=COLOR_MODE)
	img = img_to_array(img)
	img = img*(1/255)
	img = img.reshape(1, *IMG_SHAPE, CHANNEL)

	# predict
	prediction = model.predict(img)
	prediction = np.argmax(prediction, axis=1)[0]
	prediction = label2char[prediction]
	if prediction not in set(label2char_801.values()):
		prediction = 'isnull'
	####################################################

	print('\n{} is {}'.format(IMG_PATH, prediction))

if __name__=="__main__":
	main()