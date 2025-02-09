# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from gc import collect
import os

def create_directory(dataset_path: str) -> str:
    '''
    

    Returns
    -------
    string.
        Reset the content in the folders for a clean restart

    '''
    collect()
    dataset_path=os.path.expanduser(dataset_path)
    Flag=False
    if not os.path.exists(f'{dataset_path}/classes'):
        os.mkdir(f'{dataset_path}/classes')
        os.mkdir(f'{dataset_path}/classes/ai')
        os.mkdir(f'{dataset_path}/classes/real')
        Flag=True

    return dataset_path, Flag
