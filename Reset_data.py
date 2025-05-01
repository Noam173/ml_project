from gc import collect
from pathlib import Path

def create_directory(path: str) -> tuple[str, bool]:
    collect()
    dataset_path = Path(path).parent.as_posix()
    new_path = Path(f"{dataset_path}/classes")
    flag = False
    if not new_path.exists():
        new_path.mkdir(parents=True, exist_ok=True)
        (new_path / "ai").mkdir(parents=True, exist_ok=True)
        (new_path / "real").mkdir(parents=True, exist_ok=True)
        flag = True

    return dataset_path, flag
