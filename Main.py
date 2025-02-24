# -*- coding: utf-8 -*-

from Data_Manipulation import preprocess_data, split_data


def main():
    path = "~/Desktop/scripts"

    data_path = split_data(f"{path}/train.csv", path)

    preprocess_data(data_path, image_size=(224, 224), batch_size=64)


if __name__ == "__main__":
    main()
