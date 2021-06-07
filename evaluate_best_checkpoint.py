import pandas as pd
import os 
import re
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from utils import GlobalWeightedAveragePooling2D, GlobalWeightedOutputAveragePooling2D, preprocess_input




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--log_dir", type=str, required=True, help="Directory of training.log")
    parser.add_argument("-ts","--training_src", type=str, required=True, help="Specify the training source")
    parser.add_argument("-da","--data_augmentation", type=bool, default=True, help="Whether train model with data augmentation or not: 1 or 0")
    parser.add_argument("-e","--evaluate_model", action='store_true', help="Evaluate valid set")
    parser.add_argument("-cm","--color_mode", type=str, required=False, default='grayscale', help="specify color_mode in Generator")
    args = parser.parse_args()
    print(args)

    WIDTH_SHIFT_RANGE = 0.05 if args.data_augmentation else 0.0 
    HEIGHT_SHIFT_RANGE = 0.05 if args.data_augmentation else 0.0 
    ZOOM_RANGE = 0.1 if args.data_augmentation else 0.0 
    BATCH_SIZE = 256
    TRAINING_SRC = args.training_src
    COLOR_MODE = args.color_mode

    # choose the best model
    base = '..'
    models = 'models'
    training_dir = os.path.join(base, models, args.log_dir)
    training_log = os.path.join(training_dir, 'training.log')
    df = pd.read_csv(training_log)
    epoch = df[['val_accuracy']].idxmax().item()
    print('The best val_accuracy:', df.loc[epoch,"val_accuracy"])
    if epoch<10:
        epoch = '0'+str(epoch)
    else:
        epoch = str(epoch)
    print(epoch)
    models = os.listdir(training_dir)
    best_model = [model for model in models if model.startswith('model.{}-'.format(epoch))][0]
    print(best_model)
    
    # evaluate the model
    if args.evaluate_model:

        best_model_path = os.path.join(training_dir, best_model)
        model = load_model(best_model_path, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

        training_data_path = os.path.join(base,'training_data',TRAINING_SRC)
        Train_Data_Genetor = ImageDataGenerator( rescale = 1/255, # value map to [0,1]
                                            validation_split = 0.2,
                                            )
        Val_Generator = Train_Data_Genetor.flow_from_directory( training_data_path ,
                                                        target_size = (96,96),
                                                        batch_size = BATCH_SIZE,
                                                        class_mode = 'categorical',
                                                        shuffle = True, 
                                                        subset = 'validation',
                                                        color_mode=COLOR_MODE,
                                                        seed=2021
                                                        )
        score = model.evaluate_generator(Val_Generator, verbose=1)
        print('loss:', score[0])
        print('accuracy:', score[1])

if __name__=="__main__":
    main()