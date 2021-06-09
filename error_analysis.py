import pickle
import pandas as pd
import numpy as np
import os 
from pprint import pprint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from utils import GlobalWeightedAveragePooling2D, GlobalWeightedOutputAveragePooling2D

MODEL_PATH = ''
TRAINING_SRC = 'combine_similar' 
LOG_DIR = 'models_{}_lr0.0001_randomDataset_rgb'.format(TRAINING_SRC)
OUT_NAME = 'error_log_{}.csv'.format(TRAINING_SRC)

BASE = '..'
BATCH_SIZE = 256

# choose the best model
def chooseBestModel(log_dir):
    models = 'models'
    training_dir = os.path.join(BASE, models, log_dir)
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
    best_model_path = os.path.join(training_dir, best_model)
    
    return best_model_path


def main():
    model_path = chooseBestModel(LOG_DIR) if not MODEL_PATH else MODEL_PATH
    model = load_model(model_path, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

    training_data_path = os.path.join(BASE,'training_data',TRAINING_SRC)
    Train_Data_Genetor = ImageDataGenerator( rescale = 1/255, validation_split = 0.2)
    Val_Generator = Train_Data_Genetor.flow_from_directory( training_data_path ,
                                                    target_size = (96,96),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical',
                                                    shuffle = False, 
                                                    subset = 'validation',
                                                    color_mode='rgb' ,
                                                    seed=2021
                                                    )                                   
    char2label = Val_Generator.class_indices
    label2char = {value:key for key, value in char2label.items()}

    # with open('char2label.pkl', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(char2label, f, pickle.HIGHEST_PROTOCOL)
    # with open('label2char_879.pkl', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(label2char, f, pickle.HIGHEST_PROTOCOL)

    # score = model.evaluate_generator(Val_Generator, verbose=1)
    # print('loss:', score[0])
    # print('accuracy:', score[1])

    result = model.predict(Val_Generator,verbose=1)


    def getErrorFiles(checked_char):
        result_sel = result[Val_Generator.labels==char2label[checked_char]]
        predicted = np.argmax(result_sel, axis=1)
        checked_paths = [path for path in Val_Generator.filenames if checked_char in path]
        error_paths = [checked_paths[idx] for idx,predict in enumerate(predicted) if predict!=char2label[checked_char]]
        error_rate = len(error_paths)/len(checked_paths)
        # print('Accuracy for char {}: {:.1f}%'.format(checked_char,acc*100))
        return error_rate, error_paths

    # error_rate_path = {checked_char:getErrorFiles(checked_char) for checked_char in char2label.keys()}
    # l = list(error_rate_path.items())
    # sl = sorted(l, key=lambda t:t[1][0], reverse=True)
    # with open("error_log_file.log", "w") as log_file:
    #     pprint(sl, log_file)

    with open('label2char_801.pkl', 'rb') as f:
        label2char_801 = pickle.load(f)
    tranining_words_800 = set(label2char_801.values()) -set(['isnull'])

    logs = []
    for pred, label, filename in zip(result, Val_Generator.labels, Val_Generator.filenames):
        pred = np.argmax(pred)
        if pred==label:
            answer_is = 'correct'
        else: 
            answer_is = 'wrong'
        
        pred = label2char[pred]
        label = label2char[label]
        
        if TRAINING_SRC=="combine":
            log = (filename, pred, label, answer_is)
        elif TRAINING_SRC=="combine_similar":
            is_pred_isnull = pred not in tranining_words_800
            is_label_isnull = label not in tranining_words_800
            log = (filename, pred, label, answer_is, is_pred_isnull, is_label_isnull)
        
        logs.append(log)
    
    if TRAINING_SRC=="combine":
        columns = ['filename','pred','label','answer_is']
    elif TRAINING_SRC=="combine_similar":
        columns = ['filename','pred','label','answer_is', 'is_pred_isnull', 'is_label_isnull']
    
    df = pd.DataFrame(logs, columns=columns)
    df.to_csv(OUT_NAME, index=False)
    print('Accuracy: {:.4f}'.format(sum(df['answer_is']=='correct')/len(df['answer_is'])))

if __name__=="__main__":
    main()