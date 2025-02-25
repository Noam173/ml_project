# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pathlib import Path
from gc import collect


def create_directory(path: str) -> str:
    """


    Returns
    -------
    string.
        Reset the content in the folders for a clean restart

    """
    collect()
    dataset_path = Path(path).parent.as_posix()
    new_path = Path(f"{dataset_path}/classes")
    Flag = False
    if not new_path.exists():
        Path.mkdir(new_path)
        Path.mkdir(f"{new_path}/ai")
        Path.mkdir(f"{new_path}/real")
        Flag = True

    return dataset_path, Flag
