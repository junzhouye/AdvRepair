import csv
import os
import numpy as np
import torch


def mkdir_(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CSVLog:
    def __init__(self, file_root="./", file_name=None):
        mkdir_(file_root)
        self.log_path = os.path.join(file_root, file_name)

    def __call__(self, msg):
        print(msg)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(msg)


class TXTLog:
    def __init__(self,file_root="./",file_name=None):
        mkdir_(file_root)
        self.log_path = os.path.join(file_root, file_name)

    def __call__(self, msg):
        print(msg)
        with open(self.log_path,"a") as f:
            f.write(msg+"\n")
