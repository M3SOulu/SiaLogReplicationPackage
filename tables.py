from typing import List
from tabulate import tabulate


class Table:
    def __init__(self, headers: List[str]):
        self.table_dict = {k: [] for k in headers}

    def add(self, row: dict = None, **kwargs):
        for k, v in (kwargs if row is None else row).items():
            if k in self.table_dict:
                self.table_dict[k].append(v)
            else:
                self.table_dict[k.title()].append(v)

    def print(self):
        print(tabulate(self.table_dict, headers="keys", tablefmt="github"))

    def export(self, file_path):
        with open(file_path, "r") as file:
            file.write(tabulate(self.table_dict, headers="keys", tablefmt="latex"))

