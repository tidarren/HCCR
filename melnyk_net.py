import os
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
from utils import GlobalWeightedAveragePooling2D, GlobalWeightedOutputAveragePooling2D, preprocess_input


MODEL_FILEPATH = 'Melnyk-Net.hdf5'

def melnyk_net(input_shape=(96, 96, 1), reg=1e-3, global_average_type='GWAP', use_pretrained_weights=False, num_classes=801, include_top=True):	
	if global_average_type == 'GWAP':
		GlobalAveragePooling = GlobalWeightedAveragePooling2D(kernel_initializer='ones')
	elif global_average_type == 'GWOAP':
		GlobalAveragePooling = GlobalWeightedOutputAveragePooling2D(kernel_initializer='ones')
	else:
		GlobalAveragePooling = GlobalAveragePooling2D()

	if use_pretrained_weights:
		if global_average_type == 'GWAP':
			if not os.path.exists(MODEL_FILEPATH):
				print("\nError: 'Melnyk-Net.hdf5' not found")
				print('Please, donwload the model and place it in the current.')
				print('URL: https://drive.google.com/open?id=1s8PQo7CKpOGdo-eXwtYeweY8-yjs7RYp')
				input('\npress Enter to exit')			
				exit()

			model = load_model(MODEL_FILEPATH, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

			if include_top:
				return model
			else:
				dropout = model.get_layer('dropout_1')
				model_tranc = Model(inputs=model.input, outputs=dropout.output)

				return model_tranc

		else:
			print('pretrained weights available only for melnyk-net with gwap')
			exit()

	random_normal = RandomNormal(stddev=0.001, seed=1996) # output layer initializer

	input_ = Input(shape=input_shape)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(input_)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)


	x = GlobalAveragePooling(x)
	x = Dropout(0.5)(x)

	if include_top:
		x = Dense(units=num_classes, kernel_initializer=random_normal, activation='softmax', 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)

	model = Model(inputs=input_, outputs=x)

	return model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-ld","--log_dir", type=str, required=True, help="directory to save logs")
	parser.add_argument("-ts","--training_src", type=str, required=True, help="specify training source")
	parser.add_argument("-lr","--learning_rate", type=float, required=False, default=0.0001)
	parser.add_argument("-ep","--epochs", type=int, required=False, default=100)
	parser.add_argument("-de","--decay_epoch", type=str, required=False, default='', help="choose epoch to decay lr, e.g. '15,50' ")
	parser.add_argument("-bs","--batch_size", type=int, required=False, default=256)
	parser.add_argument("-wsr","--width_shift_range", type=float, required=False, default=0.05, help="Setting for Data Augmentation in Generator")
	parser.add_argument("-hsr","--height_shift_range", type=float, required=False, default=0.05, help="Setting for Data Augmentation in Generator")
	parser.add_argument("-zr","--zoom_range", type=float, required=False, default=0.1, help="Setting for Data Augmentation in Generator")
	parser.add_argument("-rr","--rotation_range", type=float, required=False, default=0, help="Setting for Data Augmentation in Generator")
	parser.add_argument("-cm","--color_mode", type=str, required=False, default='rgb', help="specify color_mode in Generator")
	args = parser.parse_args()

	LOG_DIR = args.log_dir 
	TRAINING_SRC = args.training_src #result or select or combine_similar
	LEARNING_RATE = args.learning_rate
	EPOCHS = args.epochs
	DECAY_EPOCH = args.decay_epoch
	BATCH_SIZE = args.batch_size
	WIDTH_SHIFT_RANGE = args.width_shift_range
	HEIGHT_SHIFT_RANGE = args.height_shift_range
	ZOOM_RANGE = args.zoom_range
	ROTATION_RANGE = args.rotation_range
	COLOR_MODE = args.color_mode
	NUM_CLASSES = 879 if TRAINING_SRC=='combine_similar' else 801

	if COLOR_MODE=='grayscale':
		input_shape = (96,96,1)
	elif COLOR_MODE=='rgb':
		input_shape = (96,96,3)
	
	model = melnyk_net(num_classes=NUM_CLASSES, input_shape=input_shape, use_pretrained_weights=False, include_top=True)
	
	adam = Adam(lr=LEARNING_RATE)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	# model.summary()

	image_size = (96,96)
	base = '..'
	models = 'models'
	training_data = 'training_data'
	training_data_path = os.path.join(base, training_data, TRAINING_SRC)

	models_dir = os.path.join(base,models,LOG_DIR)
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	
	# save training setting
	training_setting = os.path.join(models_dir, 'training_setting.txt')
	with open(training_setting, 'w') as f:
		print(args, file=f)
	
	# save checkpoint
	filepath = os.path.join(models_dir, 'model.{epoch:02d}-{val_loss:.2f}.hdf5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')

	lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
	early_stopping = EarlyStopping(monitor='val_loss', patience=5)

	def schedule(epoch):  
		initial_lr = K.get_value(model.optimizer.lr)
		if not DECAY_EPOCH:
			return initial_lr
		decay_epochs = DECAY_EPOCH.split(',')
		decay_epochs = [int(epoch) for epoch in decay_epochs]
		if epoch in decay_epochs:
			return initial_lr * 0.1
		return initial_lr

	lr_scheduler = LearningRateScheduler(schedule, verbose=1)
	csv_logger = CSVLogger('{}/training.log'.format(models_dir), append=False)
	callbacks_list = [checkpoint, lr_scheduler, lr_reducer, csv_logger, early_stopping]

	Train_Data_Genetor = ImageDataGenerator( rescale = 1/255, 
                                         validation_split = 0.2,
                                         width_shift_range = WIDTH_SHIFT_RANGE,   
                                         height_shift_range = HEIGHT_SHIFT_RANGE,
                                         zoom_range = ZOOM_RANGE,  
										 rotation_range = ROTATION_RANGE,
                                         horizontal_flip = False,
										 )
	Train_Generator = Train_Data_Genetor.flow_from_directory( training_data_path ,
                                                          target_size = image_size,
                                                          batch_size = BATCH_SIZE, 
                                                          class_mode = 'categorical',
                                                          shuffle = True, 
                                                          subset = 'training',
														  color_mode = COLOR_MODE
														  )
	Val_Generator = Train_Data_Genetor.flow_from_directory( training_data_path ,
                                                    target_size = image_size,
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical',
                                                    shuffle = True, 
                                                    subset = 'validation',
													color_mode = COLOR_MODE,
													seed=2021
													)
	History = model.fit_generator( Train_Generator,
                   steps_per_epoch = Train_Generator.samples//BATCH_SIZE,
                   validation_data = Val_Generator,
                   validation_steps = Val_Generator.samples//BATCH_SIZE,
                   epochs = EPOCHS,
				   callbacks=callbacks_list)

	score = model.evaluate_generator(Val_Generator, verbose=1)
	print('loss:', score[0])
	print('accuracy:', score[1])

if __name__ == '__main__':
	main()