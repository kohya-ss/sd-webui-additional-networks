import csv
from io import StringIO


def split_path_list(path_list: str) -> list[str]:
    pl = []
    with StringIO() as f:
        f.write(path_list)
        f.seek(0)
        for r in csv.reader(f):
            pl += r
    return pl
