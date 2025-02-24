# -*- coding: utf-8 -*-

from Data_Manipulation import split_data


def main():
    path = "~/scripts"

    split_data(f"{path}/train.csv", path)


if __name__ == "__main__":
    main()
