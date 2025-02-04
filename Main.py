# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from Reset_data import create_directory
from Data_Manipulation import prepare_data



def main():
    path='data'
    train_path, val_path = create_directory(path)

    prepare_data('data/train.csv', 'data/test.csv')
    
if __name__=='__main__':
    main()
