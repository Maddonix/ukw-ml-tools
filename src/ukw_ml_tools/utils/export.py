from pathlib import Path


def rename_path_if_exists(path: Path):
    suffix = path.suffix
    name = path.with_suffix("").name
    count = 0

    while path.exists():
        path = path.with_name(name + "_" + str(count)).with_suffix(suffix)
        count += 1

    return path
