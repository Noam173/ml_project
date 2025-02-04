# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import shutil
from gc import collect
import os
from glob import glob

def create_directory(dataset_path: str) -> str:
    '''
    

    Returns
    -------
    string.
        Reset the content in the folders for a clean restart

    '''
    collect()
    if (os.path.exists(dataset_path)):

        csv=glob(f'{dataset_path}/*.csv')
        csv.remove(f'{dataset_path}/test.csv')
        csv.remove(f'{dataset_path}/train.csv')
        
        list=[f'{dataset_path}/train/',f'{dataset_path}/val/']
        for x in list:
            shutil.rmtree(x, ignore_errors=True)
        for x in csv:
            os.remove(x)

        print(f'Data sources: data/{os.listdir(dataset_path)}')
        os.mkdir(f'{dataset_path}/train')
        os.mkdir(f'{dataset_path}/val')

    return f'{dataset_path}/train', f'{dataset_path}/val'