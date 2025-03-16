from gc import collect
from pathlib import Path


def create_directory(path: str) -> str:
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
