from pathlib import Path


def create_directory(path: str) -> tuple[str, bool]:
    dataset_path = Path(path).parent
    new_path = dataset_path / "classes"
    flag = False
    if not new_path.exists():
        (new_path / "ai").mkdir(parents=True)
        (new_path / "real").mkdir(parents=True)
        flag = True

    return dataset_path, flag
