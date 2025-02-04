import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np

def prepare_data(train_path_csv: str, test_path_csv: str) -> None:
    train=pd.read_csv(train_path_csv)
    test=pd.read_csv(test_path_csv)
    
    for i in train.file_name:
        path=f'data/train_data/{i}'
        img=cv2.imread(path)
        img=img.astype(np.float32)/255.0
        cv2.imwrite(path, img)

    for i in test.id:
        path=f'data/test/{i}'
        img=cv2.imread(path)
        img=img.astype(np.float32)/255.0
        cv2.imwrite(path, img)
    
    Split_Train(train_path_csv)


def move_img(train_csv: str, val_csv: str) -> str:
    train=pd.read_csv(train_csv)
    val=pd.read_csv(val_csv)
    path='data/train_data'
    for i in train.file_name:
        shutil.copy(f'{path}/{i}', 'data/train/')

    for i in val.file_name:
        shutil.copy(f'{path}/{i}', 'data/val/')

    return 'dirs: data/train, data/val'

def Split_Train(train_path_csv: str) -> str:
    '''
    Parameters
    ----------
    train_path : string.
        the path to the original dataset's DataFrame file to prepare it for splitting.

    Returns
    -------
    None.

    '''
    
    path=pd.read_csv(train_path_csv)
    
    X = path.drop('label', axis=1) 
    Y = path.label

    Y=pd.DataFrame(Y)
    x_train, x_val, y_train, y_val=train_test_split(X, Y, test_size=0.2, stratify=Y)
    
    x_train.to_csv('data/x_train.csv',index=False)
    y_train.to_csv('data/y_train.csv',index=False) 
    
    x_val.to_csv('data/x_val.csv',index=False)
    y_val.to_csv('data/y_val.csv',index=False)

    return move_img('data/x_train.csv', 'data/x_val.csv') 
    