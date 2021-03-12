import os, sys
import glob
import time

import numpy as np
import torch
import json
import nltk
import argparse
import fnmatch
import random
import copy

from transformers.tokenization_bert import BertTokenizer

def get_json_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    return files

def main():
    file_list = get_json_file_list(data_dir)
    print(file_list)
    char2num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for file_name in file_list:
        data = json.loads(open(file_name, 'r').read())
        if (data['high'][0] == 1):
            high += 1
        else:
            middle += 1