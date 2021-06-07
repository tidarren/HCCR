# %%
import pickle
import pandas as pd
import os 
import re
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from utils import GlobalWeightedAveragePooling2D, GlobalWeightedOutputAveragePooling2D


# def main():

# parser = argparse.ArgumentParser()
# parser.add_argument("-l","--log_dir", type=str, required=True, help="Directory of training.log")
# parser.add_argument("-ts","--training_src", type=str, required=True, help="Specify the training source")
# parser.add_argument("-da","--data_augmentation", type=bool, default=True, help="Whether train model with data augmentation or not: 1 or 0")
# parser.add_argument("-cm","--color_mode", type=str, required=False, default='grayscale', help="specify color_mode in Generator")
# args = parser.parse_args()
# print(args)

BATCH_SIZE = 256
LOG_DIR = 'models_combine_lr0.0001_randomDataset' #args.log_dir
TRAINING_SRC = 'combine' #args.training_src
COLOR_MODE = 'grayscale' #args.color_mode

# choose the best model
base = '..'
models = 'models'
training_dir = os.path.join(base, models, LOG_DIR)
training_log = os.path.join(training_dir, 'training.log')
df = pd.read_csv(training_log)
epoch = df[['val_accuracy']].idxmax().item()
print('The best val_accuracy:', df.loc[epoch,"val_accuracy"])
if epoch<10:
    epoch = '0'+str(epoch)
else:
    epoch = str(epoch)
print(epoch)

# evaluate the model
models = os.listdir(training_dir)
best_model = [model for model in models if model.startswith('model.{}-'.format(epoch))][0]
print(best_model)

best_model_path = os.path.join(training_dir, best_model)
model = load_model(best_model_path, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

training_data_path = os.path.join(base,'training_data',TRAINING_SRC)
Train_Data_Genetor = ImageDataGenerator( rescale = 1/255, validation_split = 0.2)
Val_Generator = Train_Data_Genetor.flow_from_directory( training_data_path ,
                                                target_size = (96,96),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'categorical',
                                                shuffle = False, 
                                                subset = 'validation',
                                                color_mode=COLOR_MODE,
                                                seed=2021
                                                )                                   
char2label = Val_Generator.class_indices
label2char = {value:key for key, value in char2label.items()}
with open('char2label.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(char2label, f, pickle.HIGHEST_PROTOCOL)
with open('label2char.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(label2char, f, pickle.HIGHEST_PROTOCOL)
# %%
score = model.evaluate_generator(Val_Generator, verbose=1)
print('loss:', score[0])
print('accuracy:', score[1])

result = model.predict(Val_Generator,verbose=1)
import numpy as np

def getErrorFiles(checked_char):
    result_sel = result[Val_Generator.labels==char2label[checked_char]]
    predicted = np.argmax(result_sel, axis=1)
    checked_paths = [path for path in Val_Generator.filenames if checked_char in path]
    error_paths = [checked_paths[idx] for idx,predict in enumerate(predicted) if predict!=char2label[checked_char]]
    error_rate = len(error_paths)/len(checked_paths)
    # print('Accuracy for char {}: {:.1f}%'.format(checked_char,acc*100))
    return error_rate, error_paths

error_rate_path = {checked_char:getErrorFiles(checked_char) for checked_char in char2label.keys()}
l = list(error_rate_path.items())
sl = sorted(l, key=lambda t:t[1][0], reverse=True)

from pprint import pprint
with open("error_log_file.log", "w") as log_file:
    pprint(sl, log_file)
# %%
# print(len(Val_Generator.labels))
# print(Val_Generator.class_indices.keys())


    # score = model.evaluate_generator(Val_Generator, verbose=1)
    # print('loss:', score[0])
    # print('accuracy:', score[1])

# if __name__=="__main__":
#     main()
# %%
