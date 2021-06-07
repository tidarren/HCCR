import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

def plot(train, valid, kind, dir):
    plt.figure(figsize=(7,5))
    plt.title(kind)
    plt.plot(range(len(train)), train, label='train')
    plt.plot(range(len(valid)), valid, label='valid')
    # plt.xticks(range(len(train)))
    plt.legend()
    plt.show()
    dir_name = os.path.basename(os.path.normpath(dir))
    plt.savefig(os.path.join(dir, '{k}_{d}.png'.format(k=kind,d=dir_name)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--log_dir", type=str, required=True, help="Directory of training.log")
    args = parser.parse_args()
    print(args)
    base = '..'
    models = 'models'
    models_dir = os.path.join(base, models, args.log_dir)
    training_log = os.path.join(models_dir, 'training.log')
    df = pd.read_csv(training_log)
    plot(df.accuracy, df.val_accuracy, 'Accuracy', models_dir)
    plot(df.loss, df.val_loss, 'Loss',models_dir)


if __name__=="__main__":
    main()