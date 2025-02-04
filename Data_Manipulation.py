import pandas as pd
import shutil
from Reset_data import create_directory
import tensorflow as tf
from model import create_model as model

def split_data(train_path_csv: str, path) -> None:
    '''
    Parameters
    ----------
    train_path : string.
        the path to the original dataset's DataFrame file to prepare it for splitting.

    Returns
    -------
    None.

    '''
    dataset_path=create_directory(path)
    path=pd.read_csv(train_path_csv)
    
    real=path[path['label']==0]
    ai=path[path['label']==1]
        
    for i in real.file_name:
        shutil.copy(f'{dataset_path}/train_data/{i}', f'{dataset_path}/classes/real')

    for i in ai.file_name:
        shutil.copy(f'{dataset_path}/train_data/{i}', f'{dataset_path}/classes/ai')

    preprocess_data(f'{dataset_path}/classes/', f'{dataset_path}/test/')
    

def preprocess_data(train_path: str, test_path: str)

    data=tf.keras.utils.image_dataset_from_directory('/home/noam/scripts/classes/', image_size=(512,512))
    test=tf.keras.utils.image_dataset_from_directory('/home/noam/scripts/test/', image_size=(512,512))

    data=data.map(lambda x,y: (x/255,y))
    test=test.map(lambda x,y: (x/255,y))

    size=len(data)
    
    train_size=int(size*.8)
    val_size=int(size*.2+1)

    train=data.take(train_size)
    val=data.skip(train_size).take(val_size)

    model(train, val)