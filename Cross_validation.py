import argparse
import os
from tqdm import tqdm
import numpy as np
import shutil
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd
import re
import yaml

SEED = random.seed()
parser = argparse.ArgumentParser()
parser.add_argument('--num_fold', type=int, default=4, help="number of fold")
parser.add_argument('--train_img_path', type=str, default="DIP_final/datasets/images/train", help="training images path")
parser.add_argument('--valid_img_path', type=str, default="DIP_final/datasets/images/test", help="validation images path")
parser.add_argument('--train_label_path', type=str, default="DIP_final/datasets/labels/train", help="training label path")
parser.add_argument('--valid_label_path', type=str, default="DIP_final/datasets/labels/test", help="validation label path")
args = parser.parse_args()

Fold = StratifiedKFold(n_splits=args.num_fold, shuffle=True, random_state=SEED)

def df_create():
    file = []
    label = []
    label_path = []
    path = []
    for _, _, fileNames in os.walk(args.train_img_path): 
        file.extend(fileNames)
        for name in fileNames:
            if re.match("powder_uncover+", name):
                label.append(0)
            elif re.match("powder_uneven+", name):
                label.append(1)
            elif re.match("scratch+", name):
                label.append(2)
    for i in range(len(file)):
        path.append(args.train_img_path + '/' + file[i])
        label_path.append(args.train_label_path + '/' + file[i].split(".")[0] + '.txt')
    file_dict = {
        "filename": file,
        "label": label,
        "label_path": label_path,
        "path": path
    }
    df = pd.DataFrame(file_dict)
    for n, (train_index, val_index) in enumerate(Fold.split(df, df['label'])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    df.to_csv('train_fold.csv', index=False)
    
    return df


def K_fold_validation(num_fold, df):
    for fold in range(num_fold):
        train_df = df.loc[df.fold != fold].reset_index(drop=True)
        valid_df = df.loc[df.fold == fold].reset_index(drop=True)
        try:
            shutil.rmtree(f'dataset_folds_{fold}/images')
            shutil.rmtree(f'dataset_folds_{fold}/labels')
        except:
            print('No dirs')
            
        print(f"Creating {fold} fold now")    
        os.makedirs(f"DIP_final/datasets_folds_{fold}/images/train", exist_ok=True)
        os.makedirs(f"DIP_final/datasets_folds_{fold}/images/valid", exist_ok=True)
        os.makedirs(f"DIP_final/datasets_folds_{fold}/labels/train", exist_ok=True)
        os.makedirs(f"DIP_final/datasets_folds_{fold}/labels/valid", exist_ok=True)
        
        for i in tqdm(range(len(train_df))):
            row = train_df.loc[i]
            shutil.copyfile(row.path, f'DIP_final/datasets_folds_{fold}/images/train/{row.filename}')
            shutil.copyfile(row.label_path, f'DIP_final/datasets_folds_{fold}/labels/train/{row.filename.split(".")[0]}.txt')
            
        for i in tqdm(range(len(valid_df))):
            row = valid_df.loc[i]
            shutil.copyfile(row.path, f'DIP_final/datasets_folds_{fold}/images/valid/{row.filename}')
            shutil.copyfile(row.label_path, f'DIP_final/datasets_folds_{fold}/labels/valid/{row.filename.split(".")[0]}.txt')

def yaml_maker(NUM_FOLD):
    for fold in range(NUM_FOLD):
        data_yaml = dict(
            path = f'D:/yolov5-master/DIP_final/datasets_folds_{fold}',
            train = 'images/train',
            val = 'images/valid',
            names = {0:'powder_uncover', 1:'powder_uneven', 2:'scratch'}
        )
        with open(f'D:/yolov5-master/DIP_final/data_fold_{fold}.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=False)
    
if __name__ == "__main__":
    num_fold = args.num_fold
    df = df_create()
    K_fold_validation(num_fold, df)
    yaml_maker(num_fold)