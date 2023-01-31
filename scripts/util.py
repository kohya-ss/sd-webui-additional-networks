import csv
from io import StringIO
from typing import List

def split_path_list(path_list: str) -> List[str]:
    pl = []
    with StringIO() as f:
        f.write(path_list)
        f.seek(0)
        for r in csv.reader(f):
            pl += r
    return pl
