"""
Code from https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/make_bu_data.py
"""

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

input_dir = '/home/mihnea/data/VQA_DL/trainval_36/'
output_dir = '/home/mihnea/data/VQA_DL/img_embeddings/'


csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# change this depending on what files are donwloaded. Usually, these files should pertain to the adaptive features
infiles = ['trainval_resnet101_faster_rcnn_genome_36.tsv']

os.makedirs(output_dir)

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(input_dir, infile), "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodebytes(item[field].encode('ascii')), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
            np.savez_compressed(os.path.join(output_dir, str(item['image_id'])), feat=item['features'])